from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import torch as th
import torch.nn as nn
from CoderSchoolAI.Environment.Attributes import *
import numpy as np
from enum import Enum

class Block(nn.Module):
    
    def __init__(self, b_type, device: th.device = th.device('cpu'), activation_function: Callable = nn.ReLU,):
        super(Block, self).__init__()
        self.device = device
        self.to(device)
        self.b_type = b_type
        self.activation_function = activation_function if activation_function is not None else Block.empty_layer
        self.forward_connections = dict()
        """
        Each key is a forward connection for a list input. 
        Example: self.forward_connections['input1'] = some_block_class(***)
        """
    def join_block(self, block: 'Block'):
        """
        Joins blocks from this block to another block.
        """
        raise NotImplementedError('This method must be implemented in the Child Block')
    
    def forward(self, x: th.Tensor):
        """
        Forward pass through the block.
        """
        raise NotImplementedError('This method must be implemented in the Child Block')
    
    def regenerate_network(self):
        """
        This function is used to correct a network and regenerate it such that it is implemented correctly.
        """
        raise NotImplementedError('This method must be implemented in the Child Block')
    
        