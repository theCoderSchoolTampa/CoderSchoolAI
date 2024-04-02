import os
import torch as th
import pickle
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
from CoderSchoolAI.Util.data_utils import get_minibatches, dict_list_to_batch, dict_to_tensor
from CoderSchoolAI.Neural.ActorCritic.ActorCriticNetwork import ActorCritic
from CoderSchoolAI.Environment.Agent import Agent, ReplayBuffer, BasicReplayBuffer, DictReplayBuffer
from CoderSchoolAI.Environment.Shell import Shell
from CoderSchoolAI.Neural.Net import Net
from collections import defaultdict
   
def deep_q_learning(
    agent: Agent, # Actor in the Environment
    environment: Shell, # Environment which the Deep Q Network is being trained on 
    q_network: Net, # The Deep Q Network to be trained
    target_q_network: Net, # Target Deep Q Network which will be updated
    buffer: Union[BasicReplayBuffer, DictReplayBuffer],  # Replay buffer to store transitions
    num_episodes:int = 1000, # Number of episodes to train the Agent on
    max_steps_per_episode: int = 100, # Stops the Epsiodic Sampling when the number of steps per episode reaches this value
    gamma: float = 0.99, # Discount factor for future rewards
    update_target_every: int = 10,  # Number of episodes between updating the target network
    batch_size: int = 32, # Size of the data used to train the Network
    epsilon=0.9, # Starting Random action selection probability
    epsilon_decay=0.997, # Decay rate for epsilon
    stop_epsilon=0.01, # Stop probability for random action selection
    alpha=0.01, # Learning rate
    reward_norm_coef:float = 1.0,
    attributes: Union[str, Tuple[str]] = None, # attributes to be used for the Network
    optimizer= None, # Optimizer to use for the Agent
    optimizer_kwargs: Optional[Dict[str, Any]] = None, # Additional keyword arguments
    fps: int = 120, # Frames per second to run the Agent
    log_frequency: int = 10, # Num Episodes Between logs
    ) -> None:
    """
    Deep Q Learning: Reinforcement Learning with Deep Q-Network
    
     For more details, you may refer to the following resources:
    
     1. Tensorflow: https://www.tensorflow.org/agents/tutorials/0_intro_rl
     2. DeepMind Playing Atari with Deep RL: https://arxiv.org/pdf/1312.5602.pdf
     3. Papers Explained, Playing Atari with Deep RL: https://www.youtube.com/watch?v=rFwQDDbYTm4
     4. Machine Learning With Phil: https://www.youtube.com/watch?v=wc-FxNENg9U
     5. Deep Q-Learning: https://arxiv.org/pdf/1509.06461.pdf
     
     Parameters:
    - agent: Agent,  Actor in the Environment
    - environment: Shell,  Environment which the Deep Q Network is being trained on 
    - q_network: Net,  The Deep Q Network to be trained
    - target_q_network: Net,  Target Deep Q Network which will be updated
    - buffer: Union[BasicReplayBuffer, DictReplayBuffer],   Replay buffer to store transitions
    - num_episodes:int = 1000,  Number of episodes to train the Agent on
    - max_steps_per_episode: int = 100, Stops the Epsiodic Sampling when the number of steps per episode reaches this value
    - gamma: float = 0.99, Discount factor for future rewards
    - update_target_every: int = 10,  Number of episodes between updating the target network
    - batch_size: int = 32, Size of the data used to train the Network
    - epsilon=0.9, Starting Random action selection probability
    - epsilon_decay=0.997, Decay rate for epsilon
    - stop_epsilon=0.01, Stop probability for random action selection
    - alpha=0.01, Learning rate
    - attributes: Union[str, Tuple[str]] = None, attributes to be used for the Network
    - optimizer= None, # Optimizer to use for the Agent
    - optimizer_kwargs: Optional[Dict[str, Any]] = None, Additional keyword arguments
    
    """
    target_q_network.train()
    q_network.train()
    episode = 1
    optimizer = th.optim.Adam(q_network.parameters(), lr=alpha) if optimizer is None else optimizer
    cumulative_reward = 0
    num_episodes_for_logging = 0
    
    if isinstance(agent.get_actions(), dict):
        raise ValueError("The action space for Deep Q Learning cannot be of type Dict.")

    def collect_rollouts():
        nonlocal epsilon, stop_epsilon, episode, cumulative_reward, num_episodes_for_logging
        state = environment.reset(attributes)
        done = False
        step = 0
        while not buffer.size() > batch_size:
            environment.clock.tick(fps)
            if done:
                state = environment.reset(attributes)
                done = False
                step = 0
                episode+=1
                num_episodes_for_logging += 1 
                if num_episodes_for_logging % log_frequency == 0:
                    avg_reward = cumulative_reward / log_frequency
                    print(f"Episode: {episode}, Avg Reward: {avg_reward}, Epsilon: {epsilon}")
                    # Reset cumulative_reward and num_episodes_for_logging
                    cumulative_reward = 0
                    num_episodes_for_logging = 0

            # Get list of possible actions from agent
            possible_actions = agent.get_actions()

            # Convert state to tensor for feeding into the network
            if isinstance(buffer, BasicReplayBuffer):
                state_tensor = th.tensor(state, dtype=th.float32)
                state_tensor = th.unsqueeze(state_tensor, 0)
            else:
                state_tensor = dict_to_tensor(state, q_network.device)
                state_tensor = {k:v.unsqueeze(0) for k, v in state_tensor.items()}
            

            # Feed the state into the q_network to get Q-values for each action
            q_values = q_network(state_tensor)

            # Choose action with highest Q-value
            _, action_index = th.max(q_values, dim=1)
            action = possible_actions[action_index.item()]

            if np.random.uniform(0, 1) < epsilon:
                action = np.random.choice(possible_actions)
                epsilon *= epsilon_decay
                epsilon = max(epsilon, stop_epsilon)

            # Take action in the environment
            next_state, reward, done = environment.step(action, 0, attributes)
            
            reward *= reward_norm_coef
            # Training Logging
            cumulative_reward += reward
            # Store transition in the replay buffer state, action, probs, vals, reward, done
            buffer.store_memory(state, action, 0, 0, reward, done, next_state)
            # Update the state
            state = next_state
            step += 1
            environment.render_env()
                
                                                # TODO: figure out: Check if the action space is dictionary type ???
    while episode < num_episodes+1:
            # Collect samples, then perform an update on the Q-network
            collect_rollouts()
            # Sample a batch from the replay buffer
            states, actions, _, _, rewards, dones, next_states, batches = buffer.generate_batches()
            buffer.clear_memory()
            # Convert to tensors
            if not isinstance(states, dict):
                states = th.tensor(states, dtype=th.float32)
                next_states = th.tensor(next_states, dtype=th.float32)
            else:
                assert isinstance(buffer, DictReplayBuffer)
                states = dict_to_tensor(states)
                next_states = dict_to_tensor(next_states)
                
            actions = th.tensor(actions, dtype=th.int64)
            actions = actions.unsqueeze(-1)
            rewards = th.tensor(rewards, dtype=th.float32)
            dones = th.tensor(dones, dtype=th.bool)

            # Get current Q-values
            current_q_values = q_network(states)
            current_q_values_for_actions = current_q_values.gather(1, actions)
            # Get next Q-values from target network
            next_q_values = target_q_network(next_states)
            max_next_q_values, _ = next_q_values.max(dim=1)
            # Compute target Q-values
            target_q_values =  rewards + gamma * (1 - dones.float()) * max_next_q_values
            target_q_values = th.unsqueeze(target_q_values, 1)
            # Compute loss
            loss = F.mse_loss(current_q_values_for_actions, target_q_values.detach())                
            if episode % log_frequency == 0: print('Loss:', loss.item())
            # Update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the target network every `update_target_every` episodes
            if episode % update_target_every == 0:
                target_q_network = q_network.copy()

def PPO(
    agent: Agent,
    environment: Union[Shell, List[Shell]],
    actor_critic_net: ActorCritic,
    buffer: Union[BasicReplayBuffer, DictReplayBuffer],
    num_episodes: int = 1000,
    max_steps_per_episode: int = 100,
    gamma: float = 0.99,
    batch_size: int = 32,
    clip_epsilon: float = 0.2,
    alpha: float = 0.001,
    epsilon: float = 0.0001,
    entropy_coef: float = 0.001,
    critic_coef:float = 0.8,
    attributes: Union[str, Tuple[str]] = None,# attributes to be used for the Network
    optimizer: Optional[th.optim.Optimizer] = None,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ppo_epochs: int = 4,
    minibatch_size:int = 16,
    fps: int = 120,
) -> None:
    """
    Proximal Policy Optimization (PPO): Reinforcement Learning with Trust Region Optimization
    
    For more details, you may refer to the following resources:

    1. Original Paper: https://arxiv.org/pdf/1707.06347.pdf
    2. OpenAI Spinning Up: https://spinningup.openai.com/en/latest/algorithms/ppo.html
    3. PyTorch Implementations: https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py 
    
    --- Note : This implementation is based on nikhilbarhate99's implementation
    
    Parameters:
    - agent: Agent, Actor in the Environment
    - environment: Union[Shell, List[Shell]], Environment for training
    - actor_critic_net: ActorCritic, the actor-critic network to be trained
    - buffer: Union[BasicReplayBuffer, DictReplayBuffer], replay buffer to store transitions
    - num_episodes: int, number of episodes to train the agent on
    - max_steps_per_episode: int, maximum steps per episode
    - gamma: float, discount factor for future rewards
    - batch_size: int, mini-batch size for each update
    - clip_epsilon: float, epsilon value for PPO clipping
    - alpha: float, learning rate for optimizer
    - attributes: Union[str, Tuple[str]], attributes to be used for the Network
    - optimizer: Optional, optimizer for learning the policy
    - optimizer_kwargs: Optional[Dict[str, Any]], additional keyword arguments for the optimizer
    - ppo_epochs: int, number of epochs to update the policy network
    - fps: int, frames per second for environment execution
    """
    actor_critic_net.train()
    episode = 1
    assert float(batch_size // minibatch_size) == batch_size /minibatch_size, "Minibatch size must be a multiple of Batch size"
    optimizer = th.optim.Adam(actor_critic_net.parameters(), lr=alpha) if optimizer is None else optimizer
    if isinstance(agent.get_actions(), dict):
        pass

    def collect_rollouts():  #TODO: Finish Vec Env Support
        # Convert to Batch
        nonlocal episode
        state = environment.reset(attributes) if not isinstance(environment, list) else dict_list_to_batch([env.reset(attributes) for env in environment])
        
        done = False if not isinstance(environment, list) else [False for _ in range(len(environment))]
        step = 0
        while not buffer.size() > batch_size:
            environment.clock.tick(fps)
            if done:
                state = environment.reset(attributes) if not isinstance(environment, list) else dict_list_to_batch([env.reset(attributes) for env in environment])
                done = False
                step = 0
                episode+=1

            # Convert state to tensor for feeding into the network
            if not isinstance(state, dict):
                state_tensor = th.tensor(state, dtype=th.float32).to(actor_critic_net.device)
                state_tensor = th.unsqueeze(state_tensor, 0)
            else:
                assert isinstance(buffer, DictReplayBuffer)
                state_tensor = dict_to_tensor(state, actor_critic_net.device)
                state_tensor = {k:v.unsqueeze(0) for k, v in state_tensor.items()}

            # Feed the state into the Actor Critic Network
            probs, actions, vals = actor_critic_net.get_sample_and_values(state_tensor)

            #TODO: This is a temporary Solution to the batched action sampling NEEDS FIX
            a_s = actions
            if len(actions.shape) > 1: a_s = a_s.squeeze(0)
                
            # Take action in the environment
            next_state, reward, done = environment.step(a_s.cpu().numpy(), 0, attributes)
            
            # Convert to Batch
            next_state = next_state if not isinstance(environment, list) else dict_list_to_batch(next_state)

            # Store transition in the replay buffer state, action, probs, vals, reward, done
            buffer.store_memory(state, actions, probs, vals, reward, done, next_state)
            # Update the state
            state = next_state
            step += 1
            environment.render_env()
            
    def compute_advantages(rewards, vals, dones):
        """
        Computes the advantage for the Critic
        """
        advantages = th.zeros_like(th.from_numpy(rewards), dtype=th.float)
        last_advantage = 0  # Assume the value function is zero for the terminal state
        for t in range(len(rewards)-1, -1, -1),:
            delta = rewards[t] + gamma * vals[t + 1] * (1 - dones[t]) - vals[t]
            last_advantage = delta + gamma * last_advantage * (1 - dones[t])
            advantages[t] = last_advantage
        return advantages   
    
    def compute_returns(rewards, dones):
        returns = th.zeros_like(th.from_numpy(rewards),  dtype=th.float)
        last_return = 0
        for t in range(len(rewards)-1, -1, -1):
            last_return = rewards[t] + gamma * last_return * (1 - dones[t])
            returns[t] = last_return
        return returns
            
        
    while episode < num_episodes+1:
        # Collect samples, then perform an update on the Q-network
        collect_rollouts()
        # Sample a batch from the replay buffer
        states, actions, log_probs, vals, rewards, dones, next_states, _ = buffer.generate_batches()
        buffer.clear_memory()
        # Convert to tensors
        if not isinstance(states, dict):
            states = th.tensor(states, dtype=th.float32,).to(device=actor_critic_net.device)
            next_states = th.tensor(next_states, dtype=th.float32, ).to(device=actor_critic_net.device)
        
        else:
            states = dict_to_tensor(states, device=actor_critic_net.device)
            next_states = dict_to_tensor(next_states, device=actor_critic_net.device)
        
        actions = th.cat(actions, dim=1).float().to(device=actor_critic_net.device) if not isinstance(actions, dict) else dict_to_tensor(actions, device=actor_critic_net.device)
        
        log_probs = th.tensor(log_probs, dtype=th.float32, ).to(device=actor_critic_net.device)
                
        advantages = compute_advantages(rewards, vals, dones).to(actor_critic_net.device) # Compute Temporal Difference
        returns = compute_returns(rewards, dones).to(actor_critic_net.device)
        
        for _ in range(ppo_epochs):
            
            for mini_states, mini_log_probs, mini_returns, mini_advantages in get_minibatches(states, actions, log_probs, returns, advantages, batch_size // minibatch_size, minibatch_size, device=actor_critic_net.device):
                # Mini-batch trhough the network
                new_log_probs, _, new_vals = actor_critic_net.get_sample_and_values(mini_states)
                # Critic Loss <- 
                critic_loss = F.mse_loss(new_vals, mini_returns)
                # Computer Ratio of new/old probs
                ratio = (th.exp(new_log_probs)), th.exp(mini_log_probs)
                
                # Actor (Policy) Loss
                surr_1 = ratio * mini_advantages
                surr_2 = th.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * mini_advantages
                actor_loss = -th.min(surr_1, surr_2).mean()
                
                # (Total Loss)
                loss = critic_coef * critic_loss + actor_loss - entropy_coef * th.mean(-th.exp(new_log_probs) * new_log_probs)
                
                # (Update)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                
                
    
            
            
            


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