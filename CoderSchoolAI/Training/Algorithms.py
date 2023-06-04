import os
import torch as th
import pickle
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
from CoderSchoolAI.Util.data_utils import dict_to_tensor
from CoderSchoolAI.Environment.Agent import Agent, ReplayBuffer, BasicReplayBuffer, DictReplayBuffer
from CoderSchoolAI.Environment.Shell import Shell
from collections import defaultdict

def deep_q_learning(
    agent: Agent, # Actor in the Environment
    environment: Shell, # Environment which the Deep Q Network is being trained on 
    q_network: Union[th.nn.Module, Callable], # The Deep Q Network to be trained
    target_q_network: Union[th.nn.Module, Callable], # Target Deep Q Network which will be updated
    buffer: Union[BasicReplayBuffer, DictReplayBuffer],  # Replay buffer to store transitions
    num_episodes:int = 1000, # Number of episodes to train the Agent on
    max_steps_per_episode: int = 100, # 
    gamma: float = 0.99, # 
    update_target_every: int = 10,  # 
    batch_size: int = 32, # 
    ) -> None:
    """
    Deep Q Learning: Reinforcement Learning with Deep Q-Network
    
     For more details, you may refer to the following resources:
    
     1. Tensorflow: https://www.tensorflow.org/agents/tutorials/0_intro_rl
     2. DeepMind Playing Atari with Deep RL: https://arxiv.org/pdf/1312.5602.pdf
     3. Papers Explained, Playing Atari with Deep RL: https://www.youtube.com/watch?v=rFwQDDbYTm4
     4. Machine Learning With Phil: https://www.youtube.com/watch?v=wc-FxNENg9U
     5. Deep Q-Learning: https://arxiv.org/pdf/1509.06461.pdf
    
    """
    # Check if the action space is dictionary type
    if isinstance(agent.get_actions(), dict):
        raise ValueError("The action space for Deep Q Learning cannot be of type Dict.")

    for episode in range(1, num_episodes+1):
        state = environment.reset()
        done = False
        step = 0

        while not done and step < max_steps_per_episode:
            step += 1
            
            # Get list of possible actions from agent
            possible_actions = agent.get_actions()

            # Convert state to tensor for feeding into the network
            state_tensor = dict_to_tensor(state)
            state_tensor = th.unsqueeze(state_tensor, 0)
            
            # Feed the state into the q_network to get Q-values for each action
            q_values = q_network(state_tensor)

            # Choose action with highest Q-value
            _, action_index = th.max(q_values, dim=1)
            action = possible_actions[action_index.item()]

            # Take action in the environment
            next_state, reward, done = environment.step(action)

            # Store transition in the replay buffer
            buffer.store_memory(state, action, _, _, reward, done)

            # Update the state
            state = next_state

            # If the replay buffer contains enough samples, then perform an update on the Q-network
            if len(buffer.states) > batch_size:
                # Sample a batch from the replay buffer
                states, actions, _, _, rewards, dones = buffer.generate_batches()

                # Convert to tensors
                states = dict_to_tensor(states)
                actions = th.tensor(actions, dtype=th.int64)
                rewards = th.tensor(rewards, dtype=th.float32)
                dones = th.tensor(dones, dtype=th.bool)

                # Get current Q-values
                current_q_values = q_network(states)

                # Get next Q-values from target network
                next_q_values = target_q_network(states)

                # Compute target Q-values
                target_q_values = rewards + (gamma * next_q_values)

                # Compute loss
                loss = F.mse_loss(current_q_values, target_q_values.detach())
                
                # Zero gradients
                q_network.optimizer.zero_grad()

                # Backpropagation
                loss.backward()

                # Update the weights
                q_network.optimizer.step()

        # Update the target network every `update_target_every` episodes
        if episode % update_target_every == 0:
            target_q_network.load_state_dict(q_network.state_dict())

class FloatDict(defaultdict):
    def __init__(self, *args):
        super().__init__(float)
        
class QLearning:
    def __init__(self, actions, alpha=0.5, gamma=0.9, epsilon=0.9, epsilon_decay=0.995, stop_epsilon=0.01):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.stop_epsilon = stop_epsilon
        self.actions = actions
        self.q_table = defaultdict(FloatDict)

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.actions)
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.stop_epsilon)
        else:
            action = max(list(range(len(self.actions))), key=lambda x: self.q_table[state][x])
        return action

    def update_q_table(self, state, action, reward, next_state):
        old_value = self.q_table[state][action]
        next_max = max(list(range(len(self.actions))), key=lambda x: self.q_table[next_state][x])
        
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state][action] = new_value
        
    def save_q_table(self, file_name):
        """
        This function stores a Q-Table as a pickle file.

        Parameters:
        q_table (dict): Q-Table to store.
        file_name (str): Name of the file to store the Q-Table in (pickle format).
        """
        with open(file_name, 'wb') as f:
            pickle.dump(self.q_table, f)


    def load_q_table(self, file_name):
        """
        This function loads a Q-Table from a pickle file.

        Parameters:
        file_name (str): Name of the file to load the Q-Table from (pickle format).

        Returns:
        dict: Loaded Q-Table.
        """
        if not os.path.exists(file_name):
            print('Cannot find file: {}'.format(file_name))
            return
        with open(file_name, 'rb') as f:
            self.q_table = pickle.load(f)

