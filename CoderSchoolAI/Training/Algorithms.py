import torch as th
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
from CoderSchoolAI.Util.data_utils import dict_to_tensor
from CoderSchoolAI.Environment.Agent import Agent, ReplayBuffer, BasicReplayBuffer, DictReplayBuffer
from CoderSchoolAI.Environment.Shell import Shell

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

    for episode in range(num_episodes):
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
