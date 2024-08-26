import pkg_resources
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union, Callable

"""
For more information regarding Spaces, see https://gym.openai.com/docs/spaces/
"""
import pygame
import gymnasium as gym
from gymnasium.spaces import Box as BoxType
from gymnasium.spaces import Dict as DictType
from gymnasium.spaces import Discrete as DiscreteType
from gymnasium.spaces import MultiDiscrete as MultiDiscreteType
from gymnasium.spaces import MultiBinary as MultiBinaryType
from CoderSchoolAI.Environments.Attributes import ObsAttribute, ActionAttribute

SpaceType = Union[BoxType, DiscreteType, MultiDiscreteType, MultiBinaryType]


class Shell:
    """
    This class represents a shell-base for an environment.
    """

    def __init__(
        self,
        target_fps: int = 30,  # Target/Max Framerate to update the environment at.
        is_user_control: bool = False,  # Sets the environment as a user-controlled environment.
        resolution: Tuple[int, int] = (84, 84),  # Resolution of the environment
        environment_name: str = "Shell",  # Name of the environment
        verbose=False,  # Whether or not to print information to the terminal.
        console_only=False,  # Makes the environment run only as a console application.
    ):
        """Init Shell

        Args:
            target_fps: int = 30,  Target/Max Framerate to update the environment at.
            is_user_control: bool = False,  Sets the environment as a user-controlled environment.
            resolution: Tuple[int, int] = (84, 84),  Resolution of the environment
            environment_name: str = "Shell",  Name of the environment
            verbose=False,  Whether or not to print information to the terminal.
         console_only=False,  Makes the environment run only as a console application.
        """
        self.target_fps = target_fps
        self.is_user_control = is_user_control
        self.ObsAttributes = dict()
        self.ActionAttributes = dict()
        self.clock = pygame.time.Clock()
        self.resolution = resolution
        self.environment_name = environment_name
        self.verbose = verbose
        self.console_only = console_only
        pygame.init()
        if not self.console_only:
            self.screen = pygame.display.set_mode(self.resolution)
            window_logo_res = pkg_resources.resource_filename(
                "CoderSchoolAI", "Assets/CoderSchoolAI/CoderSchoolAI-Logo.png"
            )
            window_logo = pygame.image.load(window_logo_res).convert()
            pygame.display.set_icon(window_logo)
            pygame.display.set_caption(f"CoderSchoolAI: {self.environment_name}")

    def __getitem__(self, name):
        """
        Returns an instance of the Attribute class for the specified attribute.
        """
        if isinstance(name, tuple):
            return {
                n: {**self.ObsAttributes, **self.ActionAttributes}.get(name, None)
                for n in name
            }
        return {**self.ObsAttributes, **self.ActionAttributes}.get(name, None)

    def get_attribute(self, name) -> Union[
        Union[ObsAttribute, ActionAttribute],
        Dict[str, Union[ObsAttribute, ActionAttribute]],
    ]:
        """
        Gets a reference for the specified attribute.
        Returns:
        - an instance of the Attribute class for the specified attribute
        """
        return self[name]

    def get_observation_space(
        self,
    ):
        return self.ObsAttributes

    def get_action_space(
        self,
    ):
        return self.ActionAttributes

    def get_observation(self, attributes=None) -> Dict[str, np.ndarray]:
        """
        Gets an observation of the environment in the form of a dictionary and a numpy array.
        """
        if attributes is None:
            attribs = {}
            for name, attr in self.ObsAttributes.items():
                if isinstance(attr.space, BoxType) or isinstance(
                    attr.space, MultiDiscreteType
                ):
                    attribs[name] = attr.data.copy()
                elif isinstance(attr.space, DiscreteType):
                    attribs[name] = attr.data
        else:
            if isinstance(attributes, tuple):
                attribs = {}
                for name, attr in self[attributes].items():
                    if isinstance(attr.space, BoxType) or isinstance(
                        attr.space, MultiDiscreteType
                    ):
                        attribs[name] = attr.data.copy()
                    elif isinstance(attr.space, DiscreteType):
                        attribs[name] = attr.data
            else:
                attr = self[attributes]
                if isinstance(attr.space, BoxType) or isinstance(
                    attr.space, MultiDiscreteType
                ):
                    return attr.data.copy()
                elif isinstance(attr.space, DiscreteType):
                    return attr.data

        return attribs

    def register_attribute(self, attribute: Union[ObsAttribute, ActionAttribute]):
        """
        Registers a new attribute with the specified name, name and range.
        """
        if isinstance(attribute, ObsAttribute):
            self.ObsAttributes[attribute.name] = attribute
        elif isinstance(attribute, ActionAttribute):
            self.ActionAttributes[attribute.name] = attribute
        else:
            raise ValueError(
                f"You cannot register an attribute of type {type(attribute)}. The attribute must be an instance of ObsAttribute or ActionAttribute."
            )

    def update_env_attributes(self):
        """
        Updates the Environment's attributes to match the current state of the Environment.
        """
        for attr in self.ObsAttributes.values():
            attr.update_func()

    def update_attribute(self, attr_name: str, new_data: Any):
        """
        Updates the specified attribute with new data.
        """
        if attr_name in self.ObsAttributes:
            self.ObsAttributes[attr_name].update(new_data)
        elif attr_name in self.ActionAttributes:
            self.ActionAttributes[attr_name].update(new_data)
        else:
            raise ValueError(
                f"The Attribute {attr_name} is not contained in the Environment."
            )

    def reset(
        self, 
        attributes=None
    ) -> Union[Dict[str, ObsAttribute], ObsAttribute, np.ndarray]:
        """
        Resets the environment.
        Returns the Initial state of the Environment.
        """
        raise NotImplementedError(
            "This method should be implemented in a subclass, not the Base Class Shell."
        )

    def step(
        self,
        action: Union[int, np.ndarray, Dict[str, ActionAttribute]],
        d_t: float,
        attributes=None,
    ) -> Tuple[
        Union[Dict[str, ObsAttribute], ObsAttribute, np.ndarray],
        Union[int, float],
        Union[bool, np.ndarray],
    ]:
        """
        Steps the environment with the specified action.
        Returns:
            - the next observation,
            - reward,
            - done flag
        """
        raise NotImplementedError(
            "This method should be implemented in a subclass, not the Base Class Shell."
        )

    def get_current_reward(
        self,
        action: Union[int, np.ndarray, Dict[str, ActionAttribute]],
        current_state: Union[Dict[str, ObsAttribute], ObsAttribute, np.ndarray],
    ) -> Tuple[Union[int, float], bool]:
        """
        Gets the reward for the specified action for a particular state.
        Returns:
            - the reward for the specified action for a particular state,
            - the done flag for the environment.
        """
        raise NotImplementedError(
            "This method should be implemented in a subclass, not the Base Class Shell."
        )

    def update_env(
        self,
    ) -> None:
        """
        Updates the environment.
        If you are working with a PyGame Based environment, this method should call render to render the environment.
        This function should calculate d_t based on the PyGame Clock and use this to provide it to the step function.

        """
        pass

    def render_env(
        self,
    ):
        """
        Should be implemented for PyGame based environments which need rendering to the screen for visualization.
        """
        pass

    @staticmethod
    def static_render_env(env: "Shell", *args, **kwargs):
        """
        Un-Implemented Method for Students to use to customize the Rendering of the Environment.
        """
        pass
