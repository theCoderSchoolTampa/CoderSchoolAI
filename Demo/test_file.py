from CoderSchoolAI import *
from CoderSchoolAI.Environment.CoderSchoolEnvironments.SnakeEnvironment import *

def learn(snake_env, steps=10000, save_file="./QSnakeAgent.pkl", log_interval=1000):
    s = 0
    while s < steps:
        snake_env.update_env() # Update the environment in what we call a loop.
        s+=1
    snake_env.snake_agent.qlearning.save_q_table(save_file)
    
def load(snake_env, steps=10000, save_file="./QSnakeAgent.pkl"):
    s = 0
    snake_env.snake_agent.qlearning.load_q_table(save_file)
    snake_env.snake_agent.qlearning.epsilon = 0
    while s < steps:
        snake_env.update_env() # Update the environment in what we call a loop.
        s+=1
snake_env = SnakeEnv(
    target_fps=6, 
    height=8,
    width=8,
    cell_size=80,
    is_user_control=False, 
    snake_is_q_table=False,
    snake_is_search_enabled=True,
    verbose=True,
    policy_kwargs=dict( # Parameters for the Q-Learning!
        alpha=0.9,  
        gamma=0.85,
        epsilon=1,
        epsilon_decay=0.999,
        )

                     ) # Create a SnakeEnv object!
snake_env.reset() # Reset the environment!
# snake_env.snake_agent.qlearning.load_q_table("./QSnakeAgent.pkl")
learn(snake_env, steps=1000000, save_file="./QSnakeAgent.pkl")
while True: # Loop until the game is over.
    snake_env.update_env() # Update the environment in what we call a loop.