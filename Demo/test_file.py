from CoderSchoolAI import *
from CoderSchoolAI.Environment.CoderSchoolEnvironments.SnakeEnvironment import *
snake_env = SnakeEnv(target_fps=6, is_user_control=True, ) # Create a SnakeEnv object!
snake_env.reset() # Reset the environment!
while True: # Loop until the game is over.
    snake_env.update_env() # Update the environment in what we call a loop.