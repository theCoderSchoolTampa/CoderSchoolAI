import argparse
from CoderSchoolAI import *
from CoderSchoolAI.Environment.CoderSchoolEnvironments.SnakeEnvironment import *

def demo():
    snake_env = SnakeEnv(
        target_fps=6, 
        height=8,
        width=8,
        cell_size=80,
        is_user_control=True, 
        snake_is_q_table=False,
        verbose=True,
        policy_kwargs=dict(
            alpha=0.9, 
            gamma=0.85,
            epsilon=1,
            epsilon_decay=0.999,
            )

                         ) # Create a SnakeEnv object!
    snake_env.reset() # Reset the environment!
    # snake_env.snake_agent.qlearning.load_q_table("./QSnakeAgent.pkl")
    # learn(snake_env, steps=1000000, save_file="./QSnakeAgent.pkl")
    while True: # Loop until the game is over.
        snake_env.update_env() # Update the environment in what we call a loop.

def main():
    parser = argparse.ArgumentParser(prog='coderschoolai')
    subparsers = parser.add_subparsers(dest='command')
    
    demo_parser = subparsers.add_parser('demo')

    args = parser.parse_args()

    if args.command == 'demo':
        demo()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()