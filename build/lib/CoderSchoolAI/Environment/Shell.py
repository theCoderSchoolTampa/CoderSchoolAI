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
SpaceType = Union[BoxType, DiscreteType, MultiDiscreteType, MultiBinaryType]

class Shell:
    """
    This class represents a shell-base for an environment.
    """
    def __init__(self,
                 target_fps: int = 30, # Target/Max Framerate to update the environment at.
                 is_user_control: bool = False, # Sets the environment as a user-controlled environment.
                 resolution: Tuple[int, int] = (84, 84), # Resolution of the environment
                 environment_name: str = "Shell", # Name of the environment
                 ):
        self.target_fps = target_fps
        self.is_user_control = is_user_control
        self.ObsAttributes = dict()
        self.ActionAttributes = dict()
        self.clock = pygame.time.Clock()
        self.resolution = resolution
        self.environment_name = environment_name
        pygame.init()
        self.screen = pygame.display.set_mode(self.resolution)
        pygame.display.set_caption(f"CoderSchoolAI: {self.environment_name}")
        
    def __getitem__(self, name):
        """ 
        Returns an instance of the Attribute class for the specified attribute.
        """
        return {**self.ObsAttributes, **self.ActionAttributes}.get(name, None)

    def get_attribute(self, name):
        """
        Gets a reference for the specified attribute.
        Returns:
        - an instance of the Attribute class for the specified attribute
        """
        return self[name]
    
    def get_observation(self) -> Dict[str, np.ndarray]:
        """
        Gets an observation of the environment in the form of a dictionary and a numpy array.
        """
        attribs = {}
        for name, attr in self.ObsAttributes.items():
            if isinstance(attr.space, BoxType) or isinstance(attr.space, MultiDiscreteType):
                attribs[name] = attr.data.copy()
            elif isinstance(attr.space, DiscreteType):
                attribs[name] = attr.data
    
    def register_attribute(self, attribute: Union[ObsAttribute, ActionAttribute]):
        """
        Registers a new attribute with the specified name, name and range.
        """
        if isinstance(attribute, ObsAttribute):
            self.ObsAttributes[attribute.name] = attribute
        elif isinstance(attribute, ActionAttribute):
            self.ActionAttributes[attribute.name] = attribute
        else:
            raise ValueError(f"You cannot register an attribute of type {type(attribute)}. The attribute must be an instance of ObsAttribute or ActionAttribute.")
    
    def update_env_attributes(self):
        """
        Updates the Environment's attributes to match the current state of the Environment.
        """
        for attr in self.ObsAttributes.values():
            attr.update_func()

    def update_attribute(self, name, new_data):
        """
        Updates the specified attribute with new data.
        """
        if name in self.ObsAttributes:
            self.ObsAttributes[name].update(new_data)
        elif name in self.ActionAttributes:
            self.ActionAttributes[name].update(new_data)
        else:
            raise ValueError(f"The Attribute {name} is not contained in the Environment.")
    
    def reset(self,) -> Union[Dict[str, ObsAttribute], ObsAttribute, np.ndarray]:
        """
        Resets the environment.
        Returns the Initial state of the Environment.
        """
        raise NotImplementedError("This method should be implemented in a subclass, not the Base Class Shell.")
    
    def step(self, action: Union[int, np.ndarray, Dict[str, ActionAttribute]], d_t:float) -> Tuple[Union[Dict[str, ObsAttribute], ObsAttribute, np.ndarray], Union[int, float], Union[bool, np.ndarray]]:
        """
        Steps the environment with the specified action.
        Returns:
            - the next observation, 
            - reward, 
            - done flag
        """
        raise NotImplementedError("This method should be implemented in a subclass, not the Base Class Shell.")
    
    def get_current_reward(self, 
                           action: Union[int, np.ndarray, Dict[str, ActionAttribute]], 
                           current_state: Union[Dict[str, ObsAttribute], ObsAttribute, np.ndarray],
                           ) -> Tuple[Union[int, float], bool]:
        """
        Gets the reward for the specified action for a particular state.
        Returns: 
            - the reward for the specified action for a particular state, 
            - the done flag for the environment.
        """
        raise NotImplementedError("This method should be implemented in a subclass, not the Base Class Shell.") 
    
    def update_env(self, ) -> None:
        """
        Updates the environment. 
        If you are working with a PyGame Based environment, this method should call render to render the environment.
        This function should calculate d_t based on the PyGame Clock and use this to provide it to the step function.
        
        """
        pass

    def render_env(self,):
        """
        Should be implemented for PyGame based environments which need rendering to the screen for visualization.
        """
        pass
    
    @staticmethod
    def static_render_env(env: 'Shell', *args, **kwargs):
        """
        Un-Implemented Method for Students to use to customize the Rendering of the Environment.
        """
        pass
    