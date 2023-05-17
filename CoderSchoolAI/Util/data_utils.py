from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import torch as th
import numpy as np
from CoderSchoolAI.Environment.Attributes import Attribute

def dict_to_tensor(dict_input, device) -> Dict[str, th.Tensor]:
    """
    Convert a dictionary of numpy arrays to PyTorch tensors, or an Attribute Dict to PyTorch tensors
    
    """
    tensor_dict = {}
    for key, value in dict_input.items():
        # Convert numpy arrays to PyTorch tensors
        if isinstance(value, Attribute):
            value = value.data
        if isinstance(value, np.ndarray):
            tensor_dict[key] = th.from_numpy(value).float().to(device)

    return tensor_dict


def euclidean_distance(End: Tuple[float, float], Start: Tuple[float, float]) -> float:
    """
    Calculates the distance between the Start Point and the End Point
    """
    return np.sqrt((End[0] - Start[0])**2 + (End[1] - Start[1])**2)

distance = euclidean_distance # Some kids would like to use this