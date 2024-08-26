import os
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from CoderSchoolAI.Environments.Attributes import *
import numpy as np
from enum import Enum

from CoderSchoolAI.Neural.Net import Net
from CoderSchoolAI.Util.neural_utils import generate_sequential, sample_distrobution


class ActorCritic(nn.Module):
    class Actor(nn.Module):
        def __init__(
            self,
            action_space: Union[ActionAttribute, Dict[str, ActionAttribute]],
            net_arch: list,
            activation_fnc=nn.ReLU,
            device="cuda",
        ):
            super(ActorCritic.Actor, self).__init__()
            self.device = device
            assert type(action_space.space) in [
                DiscreteType,
                MultiDiscreteType,
                BoxType,
            ]
            self.action_space = action_space
            if isinstance(action_space.space, BoxType):
                self._out_size = 2 * np.prod(
                    action_space.space.shape
                )  # both mean and std dev
            elif isinstance(action_space.space, DiscreteType):
                self._out_size = action_space.space.n
                self.action_outputs = [self._out_size]
            else:
                self.action_outputs = action_space.space.nvec
                self._out_size = np.sum(self.action_outputs)
            self.net_arch = net_arch + [self._out_size]
            self.net = generate_sequential(self.net_arch, activation_fnc)

        def forward(self, x):
            dist = self.net(x)
            return dist

    class Critic(nn.Module):
        def __init__(self, net_arch: list, activation_fnc=nn.ReLU, device="cuda"):
            super(ActorCritic.Critic, self).__init__()
            self._out_size = 1  # Value
            self.net_arch = net_arch + [self._out_size]
            self.net = generate_sequential(self.net_arch, activation_fnc)

        def forward(self, x):
            val = self.net(x)
            return val

    def __init__(
        self,
        observation_space: Tuple[str],
        action_space: Union[ActionAttribute, Dict[str, ActionAttribute]],
        features_extractor: nn.Module,
        net_arch: list = [256, 128, dict(vf=[64, 64], pi=[64, 64])],
        features_extractor_file: Optional[str] = None,
        train_features_extractor: bool = True,
        device="cuda",
        activation_fnc=nn.ReLU,
        actor_activation_fnc=None,
        critic_activation_fnc=None,
    ):
        """
        ActorCritic class implementing an Actor-Critic architecture for reinforcement learning.

        Parameters:
        -----------
        observation_space : Union[ObsAttribute, Dict[str, ObsAttribute]]
            The observation space describing the environment's states. It can be a single observation attribute or a dictionary of multiple observation attributes.

        action_space : Union[ActionAttribute, Dict[str, ActionAttribute]]
            The observation space describing the environment's states. It can be a single observation attribute or a dictionary of multiple observation attributes.

        features_extractor : nn.Module
            A PyTorch neural network for feature extraction from observations.
            Will be provided Observation Space if needs build. Can be a class reference or an instance of the network.

        net_arch : list
            The architecture of the neural network as a list. It may contain integers and dictionaries to specify the architecture for shared layers, actor-specific layers, and critic-specific layers.

        features_extractor_file : Optional[str] = None
            File path or name to load the trained features extractor model or left.

        train_features_extractor : bool =True
            Optional Parameter for controlling the training of a Features Extractor.

        device : str = 'cuda'
            Optional Parameter for specifying the device to run the network on.

        activation_fnc : nn.Module = nn.ReLU
            Activation function to use for the ActorCritic Network

        actor_activation_fnc : nn.Module = None
            Specify the Activation function for the actor or default to activation_fnc if left None

        critic_activation_fnc : nn.Module = None
            Specify the Activation function for the critic or default to activation_fnc if left None
        """
        super(ActorCritic, self).__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.__features_extractor_class = (
            type(features_extractor)
            if isinstance(features_extractor, nn.Module)
            else features_extractor
        )
        self.features_network = (
            self.__features_extractor_class(self.observation_space)
            if not features_extractor
            else features_extractor
        )
        assert isinstance(
            self.features_network, nn.Module
        ), "Cannot use non-torch Modules for the Features Network."

        if features_extractor_file is not None:
            self.features_network.load_state_dict(th.load(features_extractor_file))
        self.__detatch_features = not train_features_extractor
        self.__shared_arch, self.__actor_arch, self.__critic_arch = (
            ActorCritic.__get_net_arch(net_arch)
        )
        self.device = (
            th.device("cuda")
            if device == "cuda" and th.cuda.is_available()
            else th.device("cpu")
        )
        
        # Output from previous into the current
        self._features_dim = (
            self.features_network.output_size
            if isinstance(self.features_network, Net)
            else self.features_network._features_dim
        )
        self.__shared_arch = [self.features_network._features_dim] + self.__shared_arch
        self.__actor_arch, self.__critic_arch = [
            self.__shared_arch[-1]
        ] + self.__actor_arch, [self.__shared_arch[-1]] + self.__critic_arch

        if actor_activation_fnc == None:
            actor_activation_fnc = activation_fnc
        if critic_activation_fnc == None:
            critic_activation_fnc = activation_fnc
        self.shared_net = generate_sequential(self.__shared_arch)
        self.__output_activation = activation_fnc()
        self.actor = ActorCritic.Actor(
            self.action_space, self.__actor_arch, actor_activation_fnc, self.device
        )
        self.critic = ActorCritic.Critic(
            self.__critic_arch, critic_activation_fnc, self.device
        )

    def forward(self, x):
        with th.set_grad_enabled(not self.__detatch_features):
            features = self.features_network(
                x
            )  # Whether or not to use Gradients on Features Network
        features = self.__output_activation(self.shared_net(features))
        actor_output, critic_output = self.actor(features), self.critic(features)
        return actor_output, critic_output

    def sample(self, obs):
        """
        # Probabilities: [Discrete/MultiDiscrete] (batch_size, len(action_outputs), action_outputs[i] for i in action_outputs)
        # Sampled Actions: (batch_size, len(action_outputs))
        """
        actor_output, _ = self(obs)
        _action_idx = 0
        probabilities = []
        sampled_actions = []
        # Iterate through each section of Logits
        for num_actions in self.actor.action_outputs:
            # Sample action from the distribution and keep track of the probabilities
            action_probs, action_sample = sample_distrobution(
                actor_output[:, _action_idx : _action_idx + num_actions],
                self.action_space.space,
            )
            sampled_actions.append(action_sample)
            probabilities.append(action_probs)
            _action_idx += num_actions
        sampled_actions = th.stack(sampled_actions, dim=-1)
        probabilities = th.cat(probabilities, dim=1)
        return probabilities, sampled_actions

    def get_sample_and_values(self, obs):
        """
        Gets Sample Action and Values associated with observations
        - Param Obs: Actor/Critic Network input tensor
        - Output[Probabilities]: [Discrete/MultiDiscrete] (batch_size, len(action_outputs), action_outputs[i] for i in action_outputs)
        - Output[Sampled Actions]: (batch_size, len(action_outputs))
        - Output[Critic Network (Val Estimation)]: (batch_size, 1)
        """
        actor_output, critic_output = self(obs)
        _action_idx = 0
        log_probabilities = []
        sampled_actions = []
        # Iterate through each section of Logits
        for num_actions in self.actor.action_outputs:
            # Sample action from the distribution and keep track of the probabilities
            action_log_probs, action_sample = sample_distrobution(
                actor_output[:, _action_idx : _action_idx + num_actions],
                self.action_space.space,
                self.action_space,
            )
            sampled_actions.append(action_sample)
            log_probabilities.append(action_log_probs)
            _action_idx += num_actions
        sampled_actions = th.stack(sampled_actions, dim=-1)
        return log_probabilities, sampled_actions, critic_output

    @staticmethod
    def __get_net_arch(net_arch) -> Tuple[list, list, list]:
        shared_arch = []
        actor_arch = []
        critic_arch = []
        for (
            layer
        ) in net_arch:  # Add each respective architecture to the correct Network List
            if isinstance(layer, int):
                shared_arch.append(layer)
            elif isinstance(layer, dict):
                if "vf" in layer:
                    critic_arch = layer["vf"]
                else:
                    raise Exception(
                        "Must Specify a valid architecture (no Critic Network found.)"
                    )
                if "pi" in layer:
                    actor_arch = layer["pi"]
                else:
                    raise Exception(
                        "Must Specify a valid architecture (no Actor Network found.)"
                    )
        return shared_arch, actor_arch, critic_arch

    def save(self, directory: str):
        th.save(self.state_dict(), os.path.join(directory, f"ActorCritic.pt"))

    def load(self, directory: str):
        self.load_state_dict(th.load(os.path.join(directory, f"ActorCritic.pt")))
