from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import torch as th
import numpy as np
from CoderSchoolAI.Environments.Attributes import Attribute


def get_minibatches(
    states: Union[th.Tensor, Dict[str, th.Tensor]],
    probs: th.Tensor,
    returns: th.Tensor,
    advantages: th.Tensor,
    n_minibatches: int,
    mb_size: int,
    device: th.device,
):
    """
    This function generates minibatches of experiences for training Policy Gradient methods.

    Parameters:
    - states (Union[th.Tensor, Dict[str, th.Tensor]]): The states encountered in the episodes.
        Could be a single tensor or a dictionary of named tensors.
    - probs (th.Tensor): The probabilities (or log probabilities) of the actions taken, under the policy.
    - returns (th.Tensor): The calculated returns (cumulative future rewards) for each state-action pair.
    - advantages (th.Tensor): The calculated advantages for each state-action pair.
    - n_minibatches (int): The number of minibatches to divide the data into.
    - mb_size (int): The size of each minibatch.
    - device (th.device): The PyTorch device on which tensors should be stored, e.g., 'cpu' or 'cuda'.

    Returns:
    - A list of tuples of minibatches:
        (mini_states, mini_probs, mini_returns, mini_advantages)
    """
    minibatches = []
    batch_indicies = th.randperm(len(probs))
    for b_idx in range(
        n_minibatches
    ):  # Iterate through the batch sequence selecting random indicies from the data tensors
        mini_idxs = batch_indicies[b_idx : b_idx + mb_size]
        mini_states = (
            states[mini_idxs].to(device)
            if not isinstance(states, dict)
            else {k: v[mini_idxs].to(device) for k, v in states.items()}
        )
        # mini_actions = actions[mini_idxs].to(device) if not isinstance(actions, dict) else {k:v[mb_idx].to(device) for k, v in actions.items() for mb_idx in mini_idxs}
        mini_probs = probs[mini_idxs].to(device)
        mini_returns = returns[mini_idxs].to(device)
        mini_advantages = advantages[mini_idxs].to(device)
        minibatches += [(mini_states, mini_probs, mini_returns, mini_advantages)]
    return minibatches


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


def dict_to_tensor(input_dict: Union[List[Dict[str, Any]], Dict[str, List[Any]]], device) -> Dict[str, th.Tensor]:
    """
    Convert a dictionary of lists or a list of dictionaries to PyTorch tensors.
    
    Args:
        input_dict: Either a list of dictionaries or a dictionary of lists.
        device: The device to move the tensors to.
    
    Returns:
        A dictionary of PyTorch tensors.
    """
    tensor_dict = {}
    if isinstance(input_dict, list):
        for key in input_dict[0].keys():
            # Convert numpy arrays to PyTorch tensors
            tensor_dict[key] = th.tensor(np.array([d[key] for d in input_dict]), dtype=th.float32, device=device)

    elif isinstance(input_dict, dict):
        for key, value in input_dict.items():
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
    return np.sqrt((End[0] - Start[0]) ** 2 + (End[1] - Start[1]) ** 2)


distance = euclidean_distance  # Some kids would like to use this
