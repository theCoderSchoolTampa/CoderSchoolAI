import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union, Callable

"""
For more information regarding Spaces, see https://gym.openai.com/docs/spaces/
"""
import gymnasium as gym
from gymnasium.spaces import (
    Box as BoxType, 
    Dict as DictType, 
    Discrete as DiscreteType, 
    MultiDiscrete as MultiDiscreteType, 
    MultiBinary as MultiBinaryType
    )

SpaceType = Union[BoxType, DiscreteType, MultiDiscreteType, MultiBinaryType]


class Attribute:

    def __init__(
        self,
        name: str,  # Specifies the name of the attribute.
        space: SpaceType,  # Defines the type of the attribute data.
        update_func: Callable = None,  # Defines the update function of the attribute data.
    ):
        """
        This class represents an attribute of an environment, such as velocity or position.
        """
        self.name = name
        self.space = space
        self.update_func = update_func
        self.data = self.sample()
        self.shape = self._get_shape()

    def _get_shape(self):
        """
        Determines the shape of the attribute data.
        """
        return np.shape(self.data)

    def update(self, data):
        """
        Updates the attribute data and its shape.
        """
        self.data = data
        self.shape = self._get_shape()

    def sample(self, distribution: np.ndarray = None):
        """
        Samples the attribute data from a normal distribution if distrobution is None. Otherwise, it samples from the provided distribution.
        - distrobution: np.ndarray = None, # Defines the distribution of the attribute data.
        """
        if distribution is None:
            return self.space.sample()
        else:
            return distribution.sample()
        
    def copy(self):
        n_attr = Attribute(self.name, self.space)
        return n_attr

    def __str__(self):
        return f"<(Attribute: '{self.name}') Space: {self.space}, Shape: {self.shape}>"
    
    def __repr__(self):
        return str(self)

"""
Attributes of an Environment can be related to either 
"""


class ObsAttribute(Attribute):
    def __init__(
        self,
        name: str,
        space: SpaceType,
        update_func: Callable = None,
    ):
        """
        Observation Attribute of an Environment.
        - name: Specifies the name of the attribute.
        - type: SpaceType, Defines the type of the attribute data.
        - specification: Defines the shape or encoding (number of classes) for the attribute data.
        - update_func: Callable = None, # Defines the update function of the attribute data.
        """
        super().__init__(name, space, update_func)


class ActionAttribute(Attribute):
    def __init__(
        self,
        name: str,
        space: SpaceType,
        update_func: Callable = None,
    ):
        """
        Action Attribute of an Environment.
        - name: Specifies the name of the attribute.
        - type: SpaceType, Defines the type of the attribute data.
        - specification: Defines the shape or encoding (number of classes) for the attribute data.
        - update_func: Callable = None, # Defines the update function of the attribute data.
        """
        super().__init__(name, space, update_func)
