import torch as th
import torch.nn as nn
import torch.nn.functional as F
from CoderSchoolAI.Environment.Attributes import SpaceType, DiscreteType, MultiDiscreteType, BoxType, MultiBinaryType

def generate_sequential(net_arch:list, activation_fnc=nn.ReLU):
    net = []
    for i in range(len(net_arch)-1):
            net.append(nn.Linear(net_arch[i], net_arch[i+1]))
            if i < len(net_arch)-2: net.append(activation_fnc()) # Append to all except for the output dim
            return nn.Sequential(*net)
        
def sample_distrobution(logits, dist_type:SpaceType, action_space=None, eps= 1e-7):
    # Sampling an action from the distribution
    if isinstance(dist_type, MultiDiscreteType) or isinstance(dist_type, DiscreteType):
        probs = F.softmax(logits, dim=-1)
        smoothed_probs = (1 - eps) * probs + eps / probs.shape[-1]  # Epsilon-soft
        dist = th.distributions.Categorical(smoothed_probs)
        action_sample = dist.sample()
        log_prob = dist.log_prob(action_sample)
    elif isinstance(dist_type, BoxType):  # Add this branch for Gaussian distribution
        output_split = logits.size(-1)//2
        mean, log_std = th.split(logits, output_split, dim=-1)
        mean = mean.view(-1, *action_space.shape)
        log_std = log_std.view(-1, *action_space.shape)
        std = th.exp(log_std) + eps # Numerical Stability
        dist = th.distributions.Normal(mean, std)
        action_sample = dist.sample()
        log_prob = dist.log_prob(action_sample).sum(dim=(-1, -2)) if len(logits.shape) > 2 else dist.log_prob(action_sample).sum(dim=(-1,))# sum over the last n-dimensions
    else:
        raise Exception(f"Unsupported Distrobution Type: {type(dist_type)}") 
    return log_prob, action_sample

