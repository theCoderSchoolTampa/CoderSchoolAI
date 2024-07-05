from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import torch as th
import torch.nn as nn
from CoderSchoolAI.Environment.Attributes import *
from CoderSchoolAI.Neural.Block import *
import numpy as np

from CoderSchoolAI.Neural.Block import Block


class DictInputBlock(Block):
    def __init__(
        self,
        in_attributes: Dict[str, Attribute],
        modules: Dict[str, nn.Module] = False,
        device: th.device = th.device("cpu"),
    ):
        """
        This is the basis for all Deep Neural Network Blocks. This block connects all input data to a corresponding block.
        - in_attribute: the Input Attribute to be trained on. This can be either a single Attribute or a dictionary of Attributes.
        - is_module_dict: if True, the input attributes will all be passed through a dictionary of modules.
        """
        super(DictInputBlock, self).__init__(
            b_type=Block.Type.INPUT, activation_function=None, device=device
        )
        self.in_attributes = in_attributes
        self._is_convolutional = False
        self._needs_flatten = True
               
        assert isinstance(in_attributes, dict)
        assert isinstance(modules, dict)
        
        for key in set(in_attributes.keys()).union(modules.keys()):
            assert key in modules, f"No matching key from in_attributes in modules: {key}"
            assert key in in_attributes, f"No matching key from modules in in_attributes: {key}"
            
        # Multi-Attribute Networks
        flatten_size = 0
        for key, attr in in_attributes.items():
            if hasattr(modules[key], "output_size"):
                    flatten_size += getattr(modules[key], "output_size")
            else:
                x = th.zeros(size=(1,) + tuple(attr.shape))
                output = modules[key](x)
                output_size = output.size()[-1]
                flatten_size += output_size
                    
        self.flatten_size = flatten_size
        self.module = th.ModuleDict(modules)
        self.to(self.device)

    def join_block(self, block: Union[Block, List[Block]], key: str = None):
        if isinstance(block, Block):
            if block.b_type == Block.Type.LINEAR or block.b_type == Block.Type.CONV:
                if self._is_mod_dict:
                    if key == None:
                        raise ValueError(
                            "When joining an Dictionary Input Block, the key cannot be None."
                        )
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
            assert (
                not self._is_mod_dict
            ), "InputBlock of Module Dict distinction cannot be joined with a FlattenBlock."
        else:
            raise ValueError("InputBlock.join_block() expects a Block object")

    def forward(self, x) -> th.Tensor:
        if isinstance(x, np.ndarray):
            x = th.from_numpy(x)
        
        assert isinstance(x, dict), f"Error: Input not of Dict[str, Tensor], Input: {x}"
        x = {k: v.to(self.device) for k, v in x.items()}

        # Collect the tensors from each sub network            
        outputs = []
        for key, val in x.items():
            outputs.append(self.module[key](val))
        
        # Concat along the feature dimension:
        output_tnsr = th.cat(outputs, dim=1)
        
        return output_tnsr

    def regenerate_network(self):
        """
        This function is used to correct a network and regenerate it such that it is implemented correctly.
        """
        pass

    def copy(self):
        return DictInputBlock(
            in_attribute=self.in_attribute,
            is_module_dict=self._is_mod_dict,
            device=self.device,
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        res = ""
        res += f"{type(self).__module__}.{type(self).__qualname__}>,\n"

        for name, module in self.named_modules():
            if isinstance(module, nn.Sequential):
                for i, m in enumerate(module):
                    res += f"  ({i}): {repr(m)}\n"
        return res
