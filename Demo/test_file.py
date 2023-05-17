from CoderSchoolAI import *
from CoderSchoolAI.Environment.CoderSchoolEnvironments.SnakeEnvironment import *
snake_env = SnakeEnv(target_fps=6, is_user_control=True, )
snake_env.reset()
while True:
    snake_env.update_env()
    print('Apple: ', snake_env.apple_position, 'Pos:', snake_env.snake_agent.body[-1])