import numpy as np
from typing import Dict, Union, Tuple
from CoderSchoolAI.Environment.Shell import Shell
from CoderSchoolAI.Environment.Attributes import ObsAttribute, ActionAttribute
from gymnasium.spaces import Discrete, Box

class SimpleGridWorld(Shell):
    def __init__(
        self,
        grid_size: int = 5,
        target_fps: int = 30,
        is_user_control: bool = False,
        verbose: bool = False,
        console_only: bool = True
    ):
        super().__init__(
            target_fps=target_fps,
            is_user_control=is_user_control,
            resolution=(grid_size * 20, grid_size * 20),
            environment_name="SimpleGridWorld",
            verbose=verbose,
            console_only=console_only
        )
        
        self.grid_size = grid_size
        self.agent_pos = [0, 0]
        self.goal_pos = [grid_size - 1, grid_size - 1]
        
        # Register attributes
        self.register_attribute(
            ObsAttribute(
                name="agent_pos",
                space=Box(low=0, high=grid_size-1, shape=(2,), dtype=np.float32),
                update_func=self.__update_agent_pos
            )
        )
        
        self.register_attribute(
            ActionAttribute(
                name="action",
                space=Discrete(4)  # 0: up, 1: right, 2: down, 3: left
            )
        )

    def __update_agent_pos(self):
        self["agent_pos"].data = np.array(self.agent_pos, dtype=np.float32)

    def reset(self, attributes=None) -> Dict[str, np.ndarray]:
        self.agent_pos = [0, 0]
        self.__update_agent_pos()
        return self.get_observation(attributes)

    def step(
        self,
        action: int,
        d_t: float,
        attributes=None,
    ) -> Tuple[Dict[str, np.ndarray], float, bool]:
        # Move agent based on action
        if action == 0:  # up
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:  # right
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        elif action == 2:  # down
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
        elif action == 3:  # left
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)

        self.__update_agent_pos()

        # Check if goal is reached
        done = (self.agent_pos == self.goal_pos)

        # Calculate reward
        if done:
            reward = 1.0
        else:
            reward = -0.01  # Small negative reward for each step

        return self.get_observation(attributes), reward, done

    def get_current_reward(self, action: int) -> Tuple[float, bool]:
        # This method is not used in this simple environment
        pass

    def render_env(self):
        if not self.console_only:
            # Implement rendering logic here if needed
            pass
        else:
            print(f"Agent position: {self.agent_pos}")

    def update_env(self):
        # This method is not needed for this simple environment
        pass
