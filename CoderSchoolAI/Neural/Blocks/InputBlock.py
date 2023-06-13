from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import torch as th
import torch.nn as nn
from CoderSchoolAI.Environment.Attributes import *
from CoderSchoolAI.Neural.Block import *
import numpy as np

from CoderSchoolAI.Neural.Block import Block

class InputBlock(Block):
    def __init__(self,
                 in_attribute:Union[Attribute, Dict[str, Attribute]], 
                 is_module_dict:bool=False, 
                 device: th.device = th.device('cpu'),
                 ):
        super(InputBlock, self).__init__(b_type=Block.Type.INPUT, activation_function=None, device=device)
        self.in_attribute = in_attribute
        self._is_convolutional = False
        self._needs_flatten = False
        self._is_mod_dict = is_module_dict
        if isinstance(in_attribute, dict):
            temp_dict_module = {}
            flatten_size = 0
            for key, attr in in_attribute.items():
                if len(attr.shape) == 1:
                    # This is a Linear Attribute
                    temp_dict_module[key] = dict(dtype=Block.Type.LINEAR, in_features=attr.shape[0], shape=attr.shape)
                    flatten_size += attr.shape[0]
                elif len(attr.shape) > 1:
                    # This is a Convolutional Attribute
                    temp_dict_module[key] = dict(dtype=Block.Type.CONV, in_features=attr.shape[0], shape=attr.shape) if attr.shape > 2 else dict(dtype=nn.Conv2d, in_channels=1,shape=attr.shape)
                    self._is_mod_dict = True
            if self._is_mod_dict:
                self.dict_modules = temp_dict_module
            self.flatten_size = flatten_size
        
        elif isinstance(in_attribute, Attribute):
            self.module = None
            if len(in_attribute.shape) == 1:
                # This is a Linear Attribute
                self.module = dict(dtype=Block.Type.LINEAR, in_features=attr.shape[0], shape=attr.shape)
                flatten_size += attr.shape[0]
            elif len(in_attribute.shape) > 1:
                # This is a Convolutional Attribute
                self.module = dict(dtype=Block.Type.CONV, in_features=attr.shape[0], shape=attr.shape) if attr.shape > 2 else dict(dtype=nn.Conv2d, in_channels=1,shape=attr.shape)
                self._is_convolutional = True
        
    
    def join_block(self, block: Union[Block, List[Block]], key:str = None):
        if isinstance(block, Block):
            if block.d_type == Block.Type.LINEAR or block.d_type == Block.Type.CONV:
                if self._is_mod_dict:
                    if key== None:
                        raise ValueError("When joining an Dictionary Input Block, the key cannot be None.")
                    self.forward_connections[key] = block
                elif self._is_convolutional:
                    if not self.module.input_shape == block.input_shape:
                        block.input_shape = self.module.input_shape
                        block.regenerate_network()
                    self.forward_connections = block
                else:
                    if not self.module.input_shape == block.input_size:
                        block.input_shape = self.module.input_shape
                        block.regenerate_network()
                    self.forward_connections = block
            if self.device != block.device:
                block.device = self.device
                block.regenerate_network()
        else:
            raise ValueError("InputBlock.join_block() expects a Block object")
    
    def forward(self, x) -> th.Tensor:
        x = x.to(self.device)
        if self._is_mod_dict:
            join_block = self.forward_connections[self.in_attribute.keys()[0]].get_join_block()
            dict_tnsr = []
            for key, block in self.forward_connections.items():
                dict_tnsr.append(block.forward(x[key]))
            th.concat(dict_tnsr, dim=1)
        else:
            if self._is_convolutional:
                if len(x.shape) < 4:
                    self.forward(x.unsqueeze(0))
                    x = self.module(x)
                    return self.forward_connections(x)
            # Linear
            if len(x.shape) < 2:
                x = self.module(x)
                return self.forward_connections(x)
        raise Exception("InputBlock.forward() expects a 2D or 3D tensor")
        
    def regenerate_network(self):
        """
        This function is used to correct a network and regenerate it such that it is implemented correctly.
        """
        pass
    
        
                
                
                
            
    
                
                    
                    
                    
        