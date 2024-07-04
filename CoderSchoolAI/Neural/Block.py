from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import torch as th
import torch.nn as nn
from CoderSchoolAI.Environment.Attributes import *
import numpy as np
from enum import Enum


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Block(nn.Module):

    class Type(Enum):
        INPUT = 0
        LINEAR = 1
        CONV = 2
        JOIN = 3
        OUTPUT = 4
        FLATTEN = 5

    def __init__(
        self,
        b_type,
        device: th.device = th.device("cpu"),
        activation_function: Callable = nn.ReLU,
    ):
        super(Block, self).__init__()
        self.device = device
        self.to(device)
        self.b_type = b_type
        self.activation_function = (
            activation_function if activation_function is not None else Identity
        )
        self.forward_connections = dict()
        """
        Each key is a forward connection for a list input. 
        Example: self.forward_connections['input1'] = some_block_class(***)
        """

    def join_block(self, block: "Block"):
        """
        Joins blocks from this block to another block.
        """
        raise NotImplementedError("This method must be implemented in the Child Block")

    def forward(self, x: th.Tensor):
        """
        Forward pass through the block.
        """
        raise NotImplementedError("This method must be implemented in the Child Block")

    def regenerate_network(self):
        """
        This function is used to correct a network and regenerate it such that it is implemented correctly.
        """
        raise NotImplementedError("This method must be implemented in the Child Block")

    def copy(self):
        """
        This function is used to copy a Deep Neural Network Block.
        """
        raise NotImplementedError("This method must be implemented in the Child Block")

    def set_device(self, device: th.device):
        self.device = device
        self.to(device)
        self.regenerate_network()

    def compare_to(self, other):
        if type(self) != type(other):
            print(f"Block type mismatch: {type(self)} vs {type(other)}")
            return
        
        for attr in ['input_size', 'output_size', 'num_hidden_layers', 'hidden_size', 'dropout']:
            if hasattr(self, attr) and hasattr(other, attr):
                if getattr(self, attr) != getattr(other, attr):
                    print(f"Attribute {attr} mismatch: {getattr(self, attr)} vs {getattr(other, attr)}")
        
        if hasattr(self, 'module') and hasattr(other, 'module'):
            self_state = self.module.state_dict()
            other_state = other.module.state_dict()
            
            for key in self_state.keys():
                if not th.allclose(self_state[key], other_state[key], rtol=1e-5, atol=1e-8):
                    print(f"Module state mismatch in {key}")
                    print(f"Self: {self_state[key]}")
                    print(f"Other: {other_state[key]}")
                    print(f"Absolute diff: {(self_state[key] - other_state[key]).abs().max().item()}")
                    print(f"Relative diff: {((self_state[key] - other_state[key]).abs() / (self_state[key].abs() + 1e-8)).max().item()}")