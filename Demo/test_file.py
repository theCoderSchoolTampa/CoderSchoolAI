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
# from CoderSchoolAI.Environment.CoderSchoolEnvironments.SnakeEnvironment import *
# from CoderSchoolAI.Environment.Attributes import *
# from CoderSchoolAI.Neural.Blocks import *
# from CoderSchoolAI.Neural.Net import *

# image = ObsAttribute(name="img", space=BoxType(-1, 1, shape=(1, 28, 28)))
# input_block = InputBlock(in_attribute=image, is_module_dict=False)

# # Define the ConvBlock which acts as a convolutional layer for processing the game state.
# # The depth of 3 represents the number of convolutional layers in this block.
# conv_block = ConvBlock(input_shape=input_block.in_attribute.space.shape, num_channels=1, depth=4)
# lin_block = LinearBlock(input_size=conv_block.output_size, output_size=conv_block.output_size/2, hidden_size=conv_block.output_size, num_hidden_layers=3, dropout=0.2)
# # Define the OutputBlock that will decide the next action to take based on the current game state.
# # The num_classes corresponds to the number of possible actions the snake can take (up, down, left, right).
# out_block = OutputBlock(input_size=conv_block.output_size, num_classes=10)

# # Initialize the network and add the blocks
# net = Net()

# net.add_block(input_block)
# net.add_block(conv_block)
# net.add_block(out_block)
# net.compile()

### How do We use this in Our Case?
"""
Here is the Basic template for Building a Neural Network for our Snake!
"""
# from CoderSchoolAI.Environment.CoderSchoolEnvironments.SnakeEnvironment import *
# from CoderSchoolAI.Environment.CoderSchoolEnvironments.FrozenLakeEnvironment import *

# from CoderSchoolAI.Neural.Blocks import *
# from CoderSchoolAI.Neural.Net import *
# from CoderSchoolAI.Neural.ActorCritic.ActorCriticNetwork import *
# import torch as th

# snake_env = SnakeEnv(width=8, height=8)
# frozen_lake_env = FrozenLakeEnv()
# input_block = InputBlock(
#     in_attribute=frozen_lake_env.get_attribute("game_state"),
#     is_module_dict=False,
# )
# # conv_block = ConvBlock(input_shape=input_block.in_attribute.space.shape,num_channels=1,depth=5,)
# flatten_size = np.prod(frozen_lake_env.get_attribute("game_state").shape)
# flat_block = FlattenBlock(flatten_size)
# lin_block = LinearBlock(flat_block.output_size, 16, num_hidden_layers=2, hidden_size=[32, 32])

# out_block = OutputBlock(input_size=lin_block.output_size, num_classes=len(frozen_lake_env.agent.get_actions()),)

# q_net = Net(name='test_ppo_net_with_game_state')
# q_net.add_block(input_block)
# q_net.add_block(flat_block)
# q_net.add_block(lin_block)
# q_net.add_block(out_block)
# q_net.compile()

# copy_net = q_net.copy()

### Running the Training ###
"""
On Policy: PPO Agents
"""
from CoderSchoolAI.Environment.CoderSchoolEnvironments.SnakeEnvironment import *
from CoderSchoolAI.Neural.Blocks import *
from CoderSchoolAI.Neural.Net import *
from CoderSchoolAI.Neural.ActorCritic.ActorCriticNetwork import *
import torch as th

snake_env = SnakeEnv(width=16, height=16)
print("attr:", snake_env.ObsAttributes)
input_block = DictInputBlock(
    in_attributes={ "game_state": snake_env["game_state"] }, 
    modules={
        "game_state": ConvBlock(input_shape=snake_env["game_state"].shape, num_channels=1, depth=5),
    },
    device="cuda"
)
lin_block = LinearBlock(
    input_size=input_block.output_size, 
    hidden_size=512, 
    use_layer_norm=True, 
    dropout=0.3, 
    activation=nn.GELU, 
    device="cuda", 
    num_hidden_layers=1, 
    output_size=512
)


# out_block = OutputBlock(input_size=conv_block.output_size, num_classes=len(snake_env.snake_agent.get_actions()),)

ppo_net = Net(name='test_ppo_net_with_game_state')
ppo_net.add_block(input_block)
ppo_net.add_block(lin_block)
ppo_net.compile()

actor_critic_net = ActorCritic(
    snake_env["game_state"], 
    snake_env["actions"], 
    ppo_net, 
    net_arch=[512, dict(vf=[128, 128, 32], pi=[64, 64])],
    activation_fnc=nn.GELU,
)
from CoderSchoolAI.Training.Algorithms import PPO
from CoderSchoolAI.Environment.Agent import BasicReplayBuffer, DictReplayBuffer
batch_size = 512

PPO(
    agent=snake_env.snake_agent,
    environment=snake_env,
    actor_critic_net=actor_critic_net,
    buffer=DictReplayBuffer(batch_size),
    alpha=0.005,
    num_episodes=10000,
    max_steps_per_episode=100,
    batch_size=128,
    minibatch_size=128,
    critic_coef=0.3,
    entropy_coef=0.05,
    ppo_epochs=1,
    reward_norm_coef=1.0,
    reward_normalization=True,
    logging_callback=lambda: f"Apples Eaten: "
)