from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import torch as th
import torch.nn as nn
from CoderSchoolAI.Environment.Attributes import *
from CoderSchoolAI.Neural.Block import *
import numpy as np

from CoderSchoolAI.Neural.Block import Block

class ConvBlock(Block):
    def __init__(self,
                 input_shape: int,
                 num_channels: int,
                 depth: int = 3,
                 disable_max_pool: bool = False,
                 desired_output_size: int = None,
                 activation: Optional[Callable] = None,
                 device: th.device = th.device('cpu'),
                 ): #TODO: Add LayerNorm option on the End of the LinearBlock.
        """
        Creates a LinearBlock for a Neural Network.
           - input_shape (Shape): The shape of the input to the block (num channels x height x width).
           - num_channels (int): The number of pixel channels of your image (for ex. RGB = 3, RGBA = 4).
           - depth (int): The number of convolutional layers in the Block. Default is 3.
           - disable_max_pool (bool): False by default, RECOMMENDED do not disable unless errors occur.
           - activation (Callable, optional): The activation function to be used for the hidden layers. Default is None, which means no activation function is applied.
           - device (torch.device, optional): The device on which the computations will be performed. Default is 'cpu'.
        """

        activation_function = activation if activation is not None else nn.ReLU
        super(ConvBlock, self).__init__(b_type=Block.Type.LINEAR, activation_function=activation_function, device=device)
        """save these as self attributes"""
        self.input_shape = input_shape
        self.input_channels = num_channels
        self.desired_output_size = desired_output_size
        self.depth = depth
        self.module = None 
        self.output_size = None
        self.disable_max_pool = disable_max_pool
        self.regenerate_network()
    
    def get_join_block(input:Block,):
        """
        Will retrieve the shallowest JoinBlock in the network.
        """
        if input.forward_connections is None:
            return None
        if input.forward_connections.d_type == Block.Type.JOIN:
            return input.forward_connections
        
        return ConvBlock.get_join_block(input.forward_connections)
            
  
    def join_block(self, block: Block, key:str = None):
        """
        Linear Blocks can be joined with other LinearBlocks, JoinBlocks, or OutputBlocks.
        
        - Throws an error if the input block is not of the types listed above.
        """
        if isinstance(block, Block):
            if block.d_type == Block.Type.LINEAR:
                if not self.output_size == block.input_size:
                    print(f"ConvBlock.join_block() expects a LinearBlock with the same input_size as the Output of this block. Rebuilding the Block's Network.")
                    block.regenerate_network()
                self.forward_connections = block
            elif block.d_type == Block.Type.JOIN or block.d_type == Block.Type.OUTPUT:
                self.forward_connections = block
            else:
                raise Exception(f"ConvBlock.join_block() expects a LinearBlock, JoinBlock, or OutputBlock. Provided {type(block)}, {block}")
        else:
            raise ValueError("ConvBlock.join_block() expects a Block object")
    
    def forward(self, x) -> th.Tensor:
        x = x.to(self.device)
        if len(x.shape) < 4: # Ensure that the input is a Matrix
            self.forward(x.unsqueeze(0))
        elif len(x.shape) > 4: # Ensure that the input is a matrix
            x = x.squeeze(0)
            return self.forward(x)
        if x.shape[1::] != self.input_shape:
            raise Exception(f"ConvBlock.forward() expects a matrix of shape Batch Size x {self.input_shape}, Provided Shape: {x.shape}")
        x = self.module(x)
        if self.forward_connections is None or self.forward_connections.d_type == Block.Type.JOIN:
            return x
        # Another Linear Block or a Output Block
        return self.forward_connections.forward(x)
        
    def regenerate_network(self):
        """
        This function is used to correct/build a network from the internal state/structure of the block.
        """
        current_depth = 0
        prev_num_filters = self.input_channels
        layers = []
        for current_depth in range(self.depth):
            num_filters = 8*2**current_depth
            layers.append(
                nn.Conv2d(prev_num_filters, num_filters, padding= 'same', device=self.device)
            )    
            layers.append(
                self.activation_function()
            )
            if current_depth > 1 and not self.disable_max_pool:
                layers.append(
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
            prev_num_filters = num_filters
        layers.append(nn.Flatten())
        x = th.zeros(self.input_shape)
        for l in layers:
            x = l(x)
        self.output_size = x.shape[1]
        if self.desired_output_size is not None:
            layers.append(nn.Linear(self.output_size, self.desired_output_size, device=self.device))
            layers.append(self.activation_function())
        
        self.module = nn.Sequential(*layers)
    
        
                
                
                
            
    
                
                    
                    
                    
        