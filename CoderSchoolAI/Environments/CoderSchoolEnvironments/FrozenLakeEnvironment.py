import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
from CoderSchoolAI.Environments.Shell import Shell, BoxType, DictType, DiscreteType, MultiDiscreteType, MultiBinaryType
from CoderSchoolAI.Environments.Attributes import ObsAttribute, ActionAttribute
from CoderSchoolAI.Environments.Agent import Agent
from CoderSchoolAI.Environments.Shell import Shell

SpaceType = Union[BoxType, DiscreteType, MultiDiscreteType, MultiBinaryType]
"""
For more information regarding Spaces, see https://gym.openai.com/docs/spaces/
"""
from enum import IntEnum


class LakeAgent(Agent):
    class LakeAction(IntEnum):
        LEFT = 0
        RIGHT = 1
        UP = 2
        DOWN = 3

    def get_actions(self):
        return list(LakeAgent.LakeAction)

    def __init__(
        self,
    ):
        self.pos = (0, 0)


class FrozenLakeEnv(Shell):
    # You can set these as per your requirements
    class GameObjects(IntEnum):
        EMPTY = 0
        PLAYER = 1
        GOAL = 2
        HOLE = 3

    def __init__(self, height=4, width=4, verbose=False):
        super().__init__(
            console_only=True,
            environment_name="FrozenLake",
            verbose=verbose,
        )  # Initialize as per your Shell API
        self.height = height
        self.width = width
        self.agent = LakeAgent()
        self.register_attribute(
            ObsAttribute(
                name="game_state",
                space=BoxType(
                    shape=(1, height, width), low=0, high=1, dtype=np.float32
                ),
                update_func=self.__update_game_state_callback,
            )
        )
        self.register_attribute(
            ActionAttribute(
                name="actions",
                space=DiscreteType(n=len(LakeAgent.LakeAction)),
            )
        )

    def reset(self, attributes=None):
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.agent.pos = (0, 0)
        self.grid[0][0] = self.GameObjects.PLAYER
        self.grid[-1][-1] = self.GameObjects.GOAL
        self.grid[1][1] = self.grid[2][2] = (
            self.GameObjects.HOLE
        )  # Adding holes for example
        return self.get_observation(attributes)

    def step(
        self,
        action: Union[int, np.ndarray, Dict[str, ActionAttribute]],
        d_t: float,
        attributes=None,
    ):
        x, y = self.agent.pos
        if action == LakeAgent.LakeAction.LEFT:
            y = max(y - 1, 0)
        elif action == LakeAgent.LakeAction.RIGHT:
            y = min(y + 1, self.width - 1)
        elif action == LakeAgent.LakeAction.UP:
            x = max(x - 1, 0)
        elif action == LakeAgent.LakeAction.DOWN:
            x = min(x + 1, self.height - 1)
        previous_pos = self.agent.pos
        self.agent.pos = (x, y)

        if self.grid[x][y] == self.GameObjects.GOAL:
            reward, done = 1.0, True
        elif self.grid[x][y] == self.GameObjects.HOLE:
            reward, done = -1.0, True
        else:
            reward, done = 0.0, False
        self.grid[previous_pos[0]][previous_pos[1]] = self.GameObjects.EMPTY
        self.grid[x][y] = self.GameObjects.PLAYER
        if not done:  # Update Vars
            for obs in self.ObsAttributes.values():
                obs.update_func()
        return self.get_observation(attributes), reward, done

    def __update_game_state_callback(self):
        self["game_state"].data = self.grid.copy().reshape(
            1, self.height, self.width
        ) * (0.333)


# The code assumes that Agent, ObsAttribute, BoxType, DiscreteType, and Shell classes are already defined.
