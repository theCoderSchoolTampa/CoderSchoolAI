import os
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import torch as th
import torch.nn as nn
from CoderSchoolAI.Environment.Attributes import *
import numpy as np
from enum import Enum

from CoderSchoolAI.Neural.Block import Block

class Net (nn.Module):
    def __init__(self, name = "basic_net", is_dict_network=False, device: th.device = th.device('cpu'), ):
        super(Net, self).__init__()
        self.device = device
        self.to(device)
        self.name = name
        self.blocks = []
        self.block_network = None
        self.is_dict_network = is_dict_network
        self.optimizer = None
        self.__compiled = False
        """
        Each key is a forward connection for a list input. 
        Example: self.forward_connections['input1'] = some_block_class(***)
        """
    def add_block(self, block: Block):
        """
        Adds a Block connection to the network.
        """
        #TODO: add features of Graph build-out.
        # if block.b_type == Block.Type.INPUT:
        #     self.blocks.append(block)
        # elif block.b_type == Block.Type.OUTPUT:
        #     self.blocks.append(block)
        # elif block.b_type == Block.Type.JOIN:
        #     self.blocks.append(block)
        # else:
        if self.__compiled:
            raise Exception("Cannot add a block to a network that has already been compiled.")
        block.set_device(self.device)
        self.blocks.append(block)

    def compile(self):
        if self.__compiled:
            raise Exception("Cannot compile a network that has already been compiled.")
        print("Compiling Network...")
        self.block_network = nn.Sequential(*self.blocks)
        for block in self.blocks:
            print("Comiled Block: ", block)
        self.__compiled = True
    
    def forward(self, x: th.Tensor):
        """
        Forward pass through the block.
        """
        if not self.__compiled:
            raise Exception("Cannot forward a network that has not been compiled.")
        return self.block_network(x)
            
    
    def copy(self):
        """
        This function is used to copy a Deep Neural Network.
        """
        net_copy = Net(is_dict_network=self.is_dict_network, device=self.device)
        for block in self.blocks:
            if block.b_type == Block.Type.INPUT:
                inblock_copy = block.copy() # Copy the Input Block.
                if block._is_mod_dict: # Block has a Network of Forward Connections.
                    # Graph Copy, Copy each individual Sub-Network at a time.
                    for key, inner_block in inblock_copy.forward_connections.items():
                        current_copy = None
                        current_inner_block = inner_block
                        while not (current_inner_block is None or ( isinstance(current_inner_block, dict) and len(current_inner_block) == 0 )): # Redundant, but while theres something to copy.
                            inner_copy = current_inner_block.copy()
                            current_copy.join_block(inner_copy) if current_copy is not None else inblock_copy.join_block(inner_copy, key)
                            current_inner_block = current_inner_block.forward_connections
                            current_copy = inner_copy
                else: # Linear Copy
                    current_copy = None
                    current_inner_block = block.forward_connections
                    while not (current_inner_block is None or ( isinstance(current_inner_block, dict) and len(current_inner_block) ==0 )): #  Redundant, but while theres something to copy.
                        inner_copy = current_inner_block.copy()
                        current_copy.join_block(inner_copy) if current_copy is not None else inblock_copy.join_block(inner_copy)
                        current_inner_block = current_inner_block.forward_connections
                        current_copy = inner_copy  # Copy inner block, connect it to the previous, repeat.
                    net_copy.add_block(inblock_copy)
            else: # LinearBlock, ConvBlock, JoinBlock, OutputBlock
                block_copy = block.copy()
                current_copy = block_copy
                current_inner_block = block.forward_connections
                while not (current_inner_block is None or ( isinstance(current_inner_block, dict) and len(current_inner_block) ==0 )): # Redundant, but while theres something to copy.
                    inner_copy = current_inner_block.copy()
                    current_copy.join_block(inner_copy)
                    current_inner_block = current_inner_block.forward_connections
                    current_copy = inner_copy
                net_copy.add_block(block_copy) 
            
        net_copy.compile()
        return net_copy
    
    def save(self, directory: str):
        th.save(self.state_dict(), os.path.join(directory, f"{self.name}.pt"))

    def load(self, directory: str):
        self.load_state_dict(th.load(os.path.join(directory, f"{self.name}.pt")))