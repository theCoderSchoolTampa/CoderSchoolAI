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
        """
        This is the basis for all Deep Neural Network Blocks. This block connects all input data to a corresponding block.
        - in_attribute: the Input Attribute to be trained on. This can be either a single Attribute or a dictionary of Attributes.
        - is_module_dict: if True, the input attributes will all be passed through a dictionary of modules.
        """
        super(InputBlock, self).__init__(b_type=Block.Type.INPUT, activation_function=None, device=device)
        self.in_attribute = in_attribute
        self._is_convolutional = False
        self._needs_flatten = False
        self._is_mod_dict = is_module_dict
        if isinstance(in_attribute, dict): # Multi-Attribute Networks
            temp_dict_module = {}
            flatten_size = 0
            for key, attr in in_attribute.items():
                if len(attr.space.shape) == 1:
                    # This is a Linear Attribute
                    temp_dict_module[key] = dict(dtype=Block.Type.LINEAR, in_features=attr.space.shape[0], shape=attr.space.shape)
                    flatten_size += attr.shape[0]
                elif len(attr.space.shape) > 1:
                    # This is a Convolutional Attribute
                    temp_dict_module[key] = dict(dtype=Block.Type.CONV, in_features=attr.space.shape[0], shape=attr.space.shape) if len(attr.space.shape) > 2 else dict(dtype=nn.Conv2d, in_channels=1,shape=attr.space.shape.unsqueeze())
                    self._is_mod_dict = True
            if self._is_mod_dict:
                self.dict_modules = temp_dict_module
            self.flatten_size = flatten_size
        
        elif isinstance(in_attribute, Attribute): # Single Attribute Networks
            self.module = None
            if len(in_attribute.space.shape) == 1:
                # This is a Linear Attribute
                self.module = dict(dtype=Block.Type.LINEAR, in_features=attr.shape[0], shape=in_attribute.space.shape)
                flatten_size += in_attribute.space.shape[0]
            elif len(in_attribute.space.shape) > 1:
                # This is a Convolutional Attribute
                self.module = dict(dtype=Block.Type.CONV, in_features=in_attribute.space.shape[0], shape=in_attribute.space.shape) if len(in_attribute.space.shape )> 2 else dict(dtype=nn.Conv2d, in_channels=1,shape=in_attribute.space.shape)
                self._is_convolutional = True
        else:
            raise ValueError("InputBlock.__init__() expects an Attribute object or a dictionary of Attribute objects with shape > 0.")
        
    
    def join_block(self, block: Union[Block, List[Block]], key:str = None):
        if isinstance(block, Block):
            if block.b_type == Block.Type.LINEAR or block.b_type == Block.Type.CONV:
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
        elif block.b_type == Block.Type.FLATTEN:
            assert not self._is_mod_dict, "InputBlock of Module Dict distinction cannot be joined with a FlattenBlock."
        else:
            raise ValueError("InputBlock.join_block() expects a Block object")
    
    def forward(self, x) -> th.Tensor:
        if isinstance(x, np.ndarray):
            x = th.from_numpy(x)
         
        x = x.to(self.device)
        if self.forward_connections is None or ( isinstance(self.forward_connections, dict) and len(self.forward_connections) == 0 ):
            return x
         
        if self._is_mod_dict: # TODO: Implement Dictionary Input Block
            dict_tnsr = []
            for key, block in self.forward_connections.items():
                dict_tnsr.append(block(x[key]))
            return th.concat(dict_tnsr, dim=1)
        else:
            if self._is_convolutional:
                if len(x.shape) < 4:
                    self.forward(x.unsqueeze(0))
                    return self.forward_connections(x)
            # Linear
            if len(x.shape) < 2:
                x.unsqueeze(0)
            return self.forward_connections(x)
        raise Exception("InputBlock.forward() expects a 2D or 3D tensor")
        
    def regenerate_network(self):
        """
        This function is used to correct a network and regenerate it such that it is implemented correctly.
        """
        pass
    
    def copy(self):
        return InputBlock(in_attribute=self.in_attribute, is_module_dict=self._is_mod_dict, device=self.device)
    
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