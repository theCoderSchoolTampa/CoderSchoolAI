from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import torch as th
import numpy as np
from CoderSchoolAI.Environment.Attributes import Attribute
from CoderSchoolAI.Neural.ActorCritic.ActorCriticNetwork import ActorCritic

def get_minibatches(states, actions, probs, vals, advantages, batch_indices):
    for batch in batch_indices:
        mini_states = states[batch] if not isinstance(states, dict) else {k:v[batch] for k, v in states.items()}
        mini_actions = actions[batch] if not isinstance(actions, dict) else {k:v[batch] for k, v in actions.items()}
        mini_probs = probs[batch]
        mini_vals = vals[batch]
        mini_advantages = advantages[batch]
        
        yield mini_states, mini_actions, mini_probs, mini_vals, mini_advantages
        
def dict_list_to_batch(n_dicts: List[Dict]) -> Dict[str, np.ndarray]:
    """
    Converts a list of Dictionaries to a single batch. Used for nvec Environments.  
    """
    batch_dict = {}
    for i_dict in n_dicts:
        for key, value in i_dict.items():
            if key not in batch_dict:
                batch_dict[key] = []
            batch_dict[key].append(value)
    
    # Convert lists to numpy arrays for each key
    for key, value_list in batch_dict.items():
        batch_dict[key] = np.array(value_list)
    
    return batch_dict

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