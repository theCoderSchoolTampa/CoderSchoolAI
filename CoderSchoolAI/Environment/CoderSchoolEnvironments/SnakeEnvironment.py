import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
'''
For more information regarding Spaces, see https://gym.openai.com/docs/spaces/
'''
import pygame
import gymnasium as gym
from gym.spaces import Box as BoxType
from gym.spaces import Dict as DictType
from gym.spaces import Discrete as DiscreteType
from gym.spaces import MultiDiscrete as MultiDiscreteType
from gym.spaces import MultiBinary as MultiBinaryType
from CoderSchoolAI.Environment.Attributes import ObsAttribute, ActionAttribute
from CoderSchoolAI.Environment.Agent import Agent
from CoderSchoolAI.Environment.Shell import Shell
SpaceType = Union[BoxType, DiscreteType, MultiDiscreteType, MultiBinaryType]
from enum import IntEnum
from collections import deque

from CoderSchoolAI.Util.data_utils import distance, euclidean_distance 
# They are the same function, some would rather use distance because it makes more sense.


class SnakeAgent(Agent):
    class SnakeAction(IntEnum):
        LEFT = 0
        RIGHT = 1
        UP = 2
        DOWN = 3
        NOACTION = 4
    DIRECTIONS = {
            SnakeAction.LEFT: (-1, 0),
            SnakeAction.RIGHT: (1, 0),
            SnakeAction.UP: (0, -1),
            SnakeAction.DOWN: (0, 1),
            SnakeAction.NOACTION: (0, 0),
        }
    def __init__(self):
        super().__init__()
        self.body = deque([(i, 0) for i in range(3)])
        self.last_action = SnakeAgent.SnakeAction.RIGHT
        self._last_removed = None
        
    def _move_snake(self, action: 'SnakeAgent.SnakeAction') -> 'SnakeAgent.SnakeAction':
        """
        Moves the snake in the new direction, or the old direction if the snake direction has not changed (No Action).
        """
        _head = self.body[-1]
        if action == SnakeAgent.SnakeAction.NOACTION or np.dot( self.DIRECTIONS[action], self.DIRECTIONS[self.last_action]) < 0:
            action = self.last_action
        _new_head = _head[0] + self.DIRECTIONS[action][0], _head[1] + self.DIRECTIONS[action][1]
        # print('Action: ', action, 'Pos:', _new_head)
        self._last_removed = self.body.popleft()
        self.body.append(_new_head)
        self.last_action = action
        return action
    
    def increment_score(self):
        """
        Increments the score of the snake.
        """
        self.body.appendleft(self._last_removed)
        
    def reset_snake(self):
        """
        Resets the snake to its starting position.
        """
        self.body = deque([(i, 0) for i in range(3)])
        self.last_action = SnakeAgent.SnakeAction.RIGHT
        self._last_removed = None
    
    def get_actions(self):
        return list(SnakeAgent.SnakeAction)
    
    def head_intersects_body(self) -> bool:
        """
        This function returns True if the head of the snake intersects with the body of the snake.
        """
        _head = self.body[-1]
        for i in range(len(self.body)-2):
            if _head[0] == self.body[i][0] and _head[1] == self.body[i][1]:
                return True
        return False

    def get_next_action(self, state):
        # Implement your logic here to return the next action based on the state
        pass

    def update(self, state, action, next_state, reward):
        """
        This is an example of how to update the state of the Snake Agent
        """
        pass

class SnakeEnv(Shell):
    #Static Variables of the SnakeEnv
    WORLD_COLOR = (0, 0, 0)
    BODY_COLOR = (0, 255, 0)
    HEAD_COLOR = (165, 42, 42)
    APPLE_COLOR = (255, 0, 0)
    
    class GameObjects(IntEnum):
        EMPTY = 0
        BODY = 1
        HEAD = 2
        APPLE = 3
        
        
    def __init__(self, 
                 target_fps=5, # Framerate at which the Game Loop will run
                 is_user_control=False, #Flag that indicates whether the user is controlling the environment or not
                 cell_size=20, # Number of pixels in a Cell
                 height=25, # Height of the Grid
                 width=25, # Width of the Grid
                 max_length_of_snake=100, # Maximum length of the snake
                 ):
        """
        - target_fps: Framerate at which the Game Loop will run
        - is_user_control: Flag that indicates whether the user is controlling the environment or not
        - cell_size: Number of pixels in a Cell
        - height: Height of the Grid
        - width: Width of the Grid
        - max_length_of_snake: Maximum length of the snake
        """
        super().__init__(target_fps, is_user_control, resolution=(height*cell_size, width*cell_size), environment_name="Snake-Env")
        self.height = height
        self.width = width
        self.cell_size = cell_size
        self.max_length_of_snake = max_length_of_snake
        self.snake_agent = SnakeAgent()
        
        # Initialize game_state attributes, internal variables, and callbacks
        self.game_state = np.zeros((height, width, 1), dtype=np.float32)
        self.__last_moving_direction = SnakeAgent.SnakeAction.RIGHT
        self.apple_position = self.spawn_new_apple()
        self._apples_consumed = 0
        self._soft_reset = False
        
        """Register the Attributes"""
        # Game State Attribute
        self.register_attribute(ObsAttribute(name="game_state", 
        # Number of parameters in the environment to be trained on x height of image x width of image. In this case we are only training on object types.
                                             space= BoxType(shape=(1, height, width), low=0, high=1, dtype=np.float32), 
                                             update_func=self.__update_game_state_callback))
        # Moving Direction Attribute
        self.register_attribute(ObsAttribute(name="moving_direction",  
        # Number of parameters in the environment to be trained on x height of image x width of image. In this case we are only training on object types.
                                             space= DiscreteType(n=4), 
                                             update_func=self.__update_game_state_callback))
        
        # Misc Class Items
        self.font = pygame.font.Font(None, 36)  # Default font for the text. 
        

    def reset(self) -> Tuple[Union[Dict[str, ObsAttribute], ObsAttribute, np.ndarray], Union[int, float], Union[bool, np.ndarray]]:
        """
        Example of how to use the Reset Function to reset the Snake Environment.
        """
        self.apple_position = self.spawn_new_apple()
        if self._soft_reset:
            self._soft_reset = False
            self._apples_consumed += 1
            
        else:
            self.snake_agent.reset_snake()
            self._apples_consumed = 0
            self.__last_moving_direction = SnakeAgent.SnakeAction.RIGHT
        
        self.update_observation_variables()
        for name, obs in self.ObsAttributes.items():
            obs.update_func()
        return self.get_observation()

    def step(self, action: Union[int, np.ndarray, Dict[str, ActionAttribute]], d_t: float):
        """
        Example of how to use the Step Function to control the Snake.
        """
        # This will update the proper moving direction of the Snake and update the game state
        self.__last_moving_direction = self.snake_agent._move_snake(action)
        # Updates the Observation Variables
        # print(action)
        reward, finished = self.get_current_reward()
        if not finished:
            self.update_observation_variables()
            for name, obs in self.ObsAttributes.items():
                obs.update_func()
        
        # Returns the new game state, reward, and whether or not the Snake has reached the goal.
        return self.get_observation(), reward, finished
        
    def get_current_reward(self) -> Tuple[Union[int, float], bool]:
        """
        Note the Return Values:
            - Reward (Float) is the returned value of the Agent's Performance.
            - Finished is a boolean (True/False) value that indicates whether or not the Snake has reached the goal.
            
        This function uses:
            - Whether or not the Apple Was Consumed,
            - The Distance from the Snake Head to the Apple,
            - The length of the Snake Body (minus the length of the snake at the Start),
            - Whether or not The Snake is Still inside of our Grid
        """
        distance_to_apple = euclidean_distance(self.snake_agent.body[-1], self.apple_position)
        apple_consumed = self.consumed_apple()
        length_of_snake = len(self.snake_agent.body) - 3 # Minus the length of the snake at the Start
        is_in_bounds = self._is_snake_in_bounds()
        head_intersects_body = self.snake_agent.head_intersects_body()
        if not is_in_bounds or head_intersects_body:
            self._soft_reset = False
            return -1, True
        if apple_consumed:
            self._soft_reset = True
            return 1, True
        """
        Here we assign rewards for different viewable attributes of the environment.
        """
        distance_penalty = -distance_to_apple / euclidean_distance((self.width, self.height), (0, 0))
        length_of_snake_reward = length_of_snake / self.max_length_of_snake
        return distance_penalty + length_of_snake_reward, False
        
    def update_env(self):
        d_t = self.clock.tick(self.target_fps) / 1000.0

        if self.is_user_control:
            action = self.get_user_action()
        else:
            action = self.snake_agent.get_next_action(self["game_state"].data)

        state, reward, finished = self.step(action, d_t)
        if finished:
            if self.consumed_apple():
                self.snake_agent.increment_score()
            self.reset()
            
        self.render_env()

    def render_env(self):
        """
        Renders the Snake Game to the screen Via PyGame.
        """
        #Fills the world with the Blank Color
        self.screen.fill(self.WORLD_COLOR)
        for i in range(self.width):  # Assuming grid_size is the number of cells
            pygame.draw.line(self.screen, (255, 255, 255), (i * self.cell_size, 0), (i * self.cell_size, self.width * self.cell_size))  # Vertical lines
        for i in range(self.width):  # Assuming grid_size is the number of cells
            pygame.draw.line(self.screen, (255, 255, 255), (0, i * self.cell_size), (self.height * self.cell_size, i * self.cell_size))  # Horizontal lines
        # Draw the game state
        for pos in self.snake_agent.body:
                rect = pygame.Rect(pos[0] * self.cell_size, pos[1] * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, self.BODY_COLOR, rect)
        
        # Draw the head of the snake
        head_position = self.snake_agent.body[-1]
        rect = pygame.Rect(head_position[0] * self.cell_size, head_position[1] * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.HEAD_COLOR, rect)
        
        # Draw the apple
        rect = pygame.Rect(self.apple_position[0] * self.cell_size, self.apple_position[1] * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.APPLE_COLOR, rect)
        
        #Draw the score
        score_text = self.font.render(f'Score: {self._apples_consumed}', True, (225, 220, 128))
        self.screen.blit(score_text, (self.width * self.cell_size - score_text.get_width() - 5, 5))  # Draw the score on the top right corner
        
        pygame.display.flip()

    def get_user_action(self) -> SnakeAgent.SnakeAction:
        """
        This is an Example of how to use the user input to control the Snake.
        """
        # Process user input and return the corresponding action
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_s]:
            return SnakeAgent.SnakeAction.DOWN
        elif keys[pygame.K_a]:
            return SnakeAgent.SnakeAction.LEFT
        elif keys[pygame.K_d]:
            return SnakeAgent.SnakeAction.RIGHT
        elif keys[pygame.K_w]:
            return SnakeAgent.SnakeAction.UP
        return SnakeAgent.SnakeAction.NOACTION
            
    
    def update_observation_variables(self):
        """
        This is an Example of how the Game State Data can be updated.
        """
        self.game_state = np.zeros((self.height, self.width, 1), dtype=np.float32)
        self.update_agent_data()
        self.game_state[self.apple_position] = int(SnakeEnv.GameObjects.APPLE) # Apple

    def update_agent_data(self):
        """
        This function will update the Snake's Game State Data relating to the Agent.
        """
        for position in self.snake_agent.body:
            self.game_state[position] = int(SnakeEnv.GameObjects.BODY) # Snake body
        self.game_state[self.snake_agent.body[-1]] = int(SnakeEnv.GameObjects.APPLE) # Snake head
        
    def __update_game_state_callback(self):
        self['game_state'].data = self.game_state.copy().transpose(2, 0, 1) / len(list(SnakeAgent.SnakeAction))
        
    def spawn_new_apple(self) -> Tuple[int, int]:
        """
        Generates an Apple at a random position.
        """
        return (np.random.randint(0, self.width - 1), np.random.randint(0, self.height - 1))
    
    def consumed_apple(self,):
        """
        Example of how to check if the Snake has eaten an Apple.
        """
        return self.apple_position[0] == self.snake_agent.body[-1][0] and self.apple_position[1] == self.snake_agent.body[-1][1]

    def _is_snake_in_bounds(self, ):
        """
        Example of how to check if the Snake has exited the Grid.
        """
        head_position = self.snake_agent.body[-1]
        return (0 <= head_position[0] < self.width) and (0 <= head_position[1] < self.height)
            
    
    