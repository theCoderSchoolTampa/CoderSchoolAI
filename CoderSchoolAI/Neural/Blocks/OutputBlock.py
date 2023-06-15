from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import torch as th
import torch.nn as nn
from CoderSchoolAI.Environment.Attributes import *
from CoderSchoolAI.Neural.Block import *

class OutputBlock(Block):
    def __init__(self,
                 input_size: int,
                 num_classes: int,
                 activation: Callable = Identity,
                 device: th.device = th.device('cpu'),
                 ): 
        """
        Creates an OutputBlock for a Neural Network.
           - input_size (int): The size of the input to the block.
           - num_classes (int): The number of classes for the output.
           - activation (Callable, optional): The activation function to be used for the hidden layers. Default is an Identity Layer, which means no activation function is applied.
           - device (torch.device, optional): The device on which the computations will be performed. Default is 'cpu'.
        """
        super(OutputBlock, self).__init__(b_type=Block.Type.OUTPUT, device=device, activation_function=activation)

        """save these as self attributes"""
        self.input_size = input_size
        self.num_classes = num_classes
        self.module = None
        
        self.regenerate_network()
    
    def forward(self, x) -> th.Tensor:
        x = x.to(self.device)
        if len(x.shape) < 2: # Ensure that the input is a vector
            self.forward(x.unsqueeze(0))
        elif len(x.shape) > 2: # Ensure that the input is a matrix
            x = x.squeeze(0)
            return self.forward(x)
        if x.shape[-1] != self.input_size:
            raise Exception(f"OutputBlock.forward() expects a matrix of shape Batch Size x {self.input_size}, Provided Shape: {x.shape}")
        x = self.module(x)
        return x
            
        
    def regenerate_network(self):
        """
        This function is used to correct/build a network from the internal state/structure of the block.
        """
        layers = []
        layers.append(nn.Linear(self.input_size, self.num_classes, device=self.device))
        layers.append(self.activation_function())
        self.module = nn.Sequential(*layers)
        
    def copy(self):
        output_copy = OutputBlock(self.input_size, self.num_classes, activation=self.activation_function, device=self.device)
        output_copy.module.load_state_dict(self.module.state_dict())
        return output_copy
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __str__(self) -> str:
        res = ""
        res+=f"{type(self).__module__}.{type(self).__qualname__}>,\n"

        for name, module in self.named_modules():
            if isinstance(module, nn.Sequential):
                for i, m in enumerate(module):
                    res += (f"  ({i}): {repr(m)}\n")
        return res