### Testing Neural Network ###
# from CoderSchoolAI import *
# from CoderSchoolAI.Environment.CoderSchoolEnvironments.SnakeEnvironment import *

# def learn(snake_env, steps=10000, save_file="./QSnakeAgent.pkl", log_interval=1000):
#     s = 0
#     while s < steps:
#         snake_env.update_env() # Update the environment in what we call a loop.
#         s+=1
#     snake_env.snake_agent.qlearning.save_q_table(save_file)
    
# def load(snake_env, steps=10000, save_file="./QSnakeAgent.pkl"):
#     s = 0
#     snake_env.snake_agent.qlearning.load_q_table(save_file)
#     snake_env.snake_agent.qlearning.epsilon = 0
#     while s < steps:
#         snake_env.update_env() # Update the environment in what we call a loop.
#         s+=1

# snake_env = SnakeEnv(
#     target_fps=6, 
#     height=8,
#     width=8,
#     cell_size=80,
#     is_user_control=False, 
#     snake_is_q_table=False,
#     snake_is_search_enabled=True,
#     verbose=True,
#     policy_kwargs=dict( # Parameters for the Q-Learning!
#         alpha=0.9,  
#         gamma=0.85,
#         epsilon=1,
#         epsilon_decay=0.999,
#         )

#                      ) # Create a SnakeEnv object!
# snake_env.reset() # Reset the environment!
# # snake_env.snake_agent.qlearning.load_q_table("./QSnakeAgent.pkl")
# learn(snake_env, steps=1000000, save_file="./QSnakeAgent.pkl")
# while True: # Loop until the game is over.
#     snake_env.update_env() # Update the environment in what we call a loop.



### Testing Neural Network ###
# from CoderSchoolAI.Environment.CoderSchoolEnvironments.SnakeEnvironment import *
# from CoderSchoolAI.Neural.Blocks import *
# from CoderSchoolAI.Neural.Net import *
# import torch as th
# snake_env = SnakeEnv(width=16, height=16)
# input_block = InputBlock(in_attribute=snake_env.get_attribute("game_state"), is_module_dict=False,)
# conv_block = ConvBlock(input_shape=input_block.in_attribute.space.shape,num_channels=1,depth=5,)
# out_block = OutputBlock(input_size=conv_block.output_size, num_classes=len(snake_env.snake_agent.get_actions()),)
# net = Net()
# net.add_block(input_block)
# net.add_block(conv_block)
# net.add_block(out_block)
# net.compile()
# input_sample = snake_env.get_attribute("game_state").sample()
# output_test = net(input_sample)
# copy_net = net.copy()
# output_copy_test = copy_net(input_sample)

### Testing Algorithms ###
# from CoderSchoolAI.Training.Algorithms import deep_q_learning
# from CoderSchoolAI.Environment.Agent import BasicReplayBuffer
# batch_size = 32
# deep_q_learning(
#     agent=snake_env.snake_agent,
#     environment=snake_env,
#     q_network=copy_net,
#     target_q_network=net,
#     buffer= BasicReplayBuffer(batch_size),
#     num_episodes=1000,
#     epsilon_decay=999,
#     max_steps_per_episode=100,
#     batch_size=batch_size,
#     alpha=0.01,
#     attributes="game_state",
# )

### Testing MSIT ###
from CoderSchoolAI.Environment.CoderSchoolEnvironments.SnakeEnvironment import *
from CoderSchoolAI.Environment.Attributes import *
from CoderSchoolAI.Neural.Blocks import *
from CoderSchoolAI.Neural.Net import *

image = ObsAttribute(name="img", space=BoxType(-1, 1, shape=(1, 28, 28)))
input_block = InputBlock(in_attribute=image, is_module_dict=False)

# Define the ConvBlock which acts as a convolutional layer for processing the game state.
# The depth of 3 represents the number of convolutional layers in this block.
conv_block = ConvBlock(input_shape=input_block.in_attribute.space.shape, num_channels=1, depth=4)
lin_block = LinearBlock(input_size=conv_block.output_size, output_size=conv_block.output_size/2, hidden_size=conv_block.output_size, num_hidden_layers=3, dropout=0.2)
# Define the OutputBlock that will decide the next action to take based on the current game state.
# The num_classes corresponds to the number of possible actions the snake can take (up, down, left, right).
out_block = OutputBlock(input_size=conv_block.output_size, num_classes=10)

# Initialize the network and add the blocks
net = Net()

net.add_block(input_block)
net.add_block(conv_block)
net.add_block(out_block)
net.compile()