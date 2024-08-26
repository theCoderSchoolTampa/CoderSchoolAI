import os
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import torch as th
import torch.nn as nn
from CoderSchoolAI.Environments.Attributes import *
import numpy as np
from enum import Enum
import pickle

from CoderSchoolAI.Neural.Block import Block


class Net(nn.Module):
    def __init__(
        self,
        name="basic_net",
        is_dict_network=False,
        device: th.device = th.device("cpu"),
    ):
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
    
    @property
    def output_size(self):
        return self._features_dim
    
    @property
    def _features_dim(self):
        assert len(self.blocks) > 0
        return self.blocks[-1].output_size
    
    def add_block(self, block: Block):
        """
        Adds a Block connection to the network.
        """
        # TODO: add features of Graph build-out.
        # if block.b_type == Block.Type.INPUT:
        #     self.blocks.append(block)
        # elif block.b_type == Block.Type.OUTPUT:
        #     self.blocks.append(block)
        # elif block.b_type == Block.Type.JOIN:
        #     self.blocks.append(block)
        # else:
        if self.__compiled:
            raise Exception(
                "Cannot add a block to a network that has already been compiled."
            )
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

    # def copy(self):
    #     """
    #     This function is used to copy a Deep Neural Network.
    #     """
    #     net_copy = Net(is_dict_network=self.is_dict_network, device=self.device)
    #     for block in self.blocks:
    #         if block.b_type == Block.Type.INPUT:
    #             inblock_copy = block.copy()  # Copy the Input Block.

    #             if block._is_mod_dict:  # Block has a Network of Forward Connections.
    #                 # Graph Copy, Copy each individual Sub-Network at a time.
    #                 for key, inner_block in inblock_copy.forward_connections.items():
    #                     current_copy = None
    #                     current_inner_block = inner_block
    #                     while not (
    #                         current_inner_block is None
    #                         or (
    #                             isinstance(current_inner_block, dict)
    #                             and len(current_inner_block) == 0
    #                         )
    #                     ):  # Redundant, but while theres something to copy.
    #                         inner_copy = current_inner_block.copy()
    #                         (
    #                             current_copy.join_block(inner_copy)
    #                             if current_copy is not None
    #                             else inblock_copy.join_block(inner_copy, key)
    #                         )
    #                         current_inner_block = (
    #                             current_inner_block.forward_connections
    #                         )
    #                         current_copy = inner_copy
                            
    #             else:  # Linear Copy
    #                 current_copy = None
    #                 current_inner_block = block.forward_connections
    #                 while not (
    #                     current_inner_block is None
    #                     or (
    #                         isinstance(current_inner_block, dict)
    #                         and len(current_inner_block) == 0
    #                     )
    #                 ):  #  Redundant, but while theres something to copy.
    #                     inner_copy = current_inner_block.copy()
    #                     (
    #                         current_copy.join_block(inner_copy)
    #                         if current_copy is not None
    #                         else inblock_copy.join_block(inner_copy)
    #                     )
    #                     current_inner_block = current_inner_block.forward_connections
    #                     current_copy = inner_copy  # Copy inner block, connect it to the previous, repeat.
    #                 net_copy.add_block(inblock_copy)
                    
    #         else:  # LinearBlock, ConvBlock, JoinBlock, OutputBlock
    #             block_copy = block.copy()
    #             current_copy = block_copy
    #             current_inner_block = block.forward_connections
    #             while not (
    #                 current_inner_block is None
    #                 or (
    #                     isinstance(current_inner_block, dict)
    #                     and len(current_inner_block) == 0
    #                 )
    #             ):  # Redundant, but while theres something to copy.
    #                 inner_copy = current_inner_block.copy()
    #                 current_copy.join_block(inner_copy)
    #                 current_inner_block = current_inner_block.forward_connections
    #                 current_copy = inner_copy
    #             net_copy.add_block(block_copy)

    #     net_copy.compile()
    #     net_copy.load_state_dict(self.state_dict())
        
    #     return net_copy
    def copy(self):
        new_net = Net(name=self.name, is_dict_network=self.is_dict_network, device=self.device)
        for block in self.blocks:
            new_net.add_block(block.copy())
        new_net.compile()
        new_net.load_state_dict(self.state_dict(), strict=True)
        return new_net
    
    def compare_networks(self, other_net):
        print("Comparing networks...")
        
        # Compare overall structure
        if len(self.blocks) != len(other_net.blocks):
            print(f"Number of blocks mismatch: {len(self.blocks)} vs {len(other_net.blocks)}")
        
        # Compare each block
        for i, (self_block, other_block) in enumerate(zip(self.blocks, other_net.blocks)):
            if type(self_block) != type(other_block):
                print(f"Block {i} type mismatch: {type(self_block)} vs {type(other_block)}")
            else:
                print(f"Comparing block {i}...")
                self_block.compare_to(other_block)
    
        # Compare state dicts
        self_state = self.state_dict()
        other_state = other_net.state_dict()
        
        if self_state.keys() != other_state.keys():
            print("State dict keys mismatch")
            print("Self keys:", self_state.keys())
            print("Other keys:", other_state.keys())
        
        for key in self_state.keys():
            if not th.allclose(self_state[key], other_state[key], rtol=1e-5, atol=1e-8):
                print(f"Mismatch in {key}")
                print(f"Self: {self_state[key]}")
                print(f"Other: {other_state[key]}")
                print(f"Absolute diff: {(self_state[key] - other_state[key]).abs().max().item()}")
                print(f"Relative diff: {((self_state[key] - other_state[key]).abs() / (self_state[key].abs() + 1e-8)).max().item()}")


    def save(self, directory: str):
        th.save(self.state_dict(), os.path.join(directory, f"{self.name}.pt"))

    def load(self, directory: str):
        self.load_state_dict(th.load(os.path.join(directory, f"{self.name}.pt")))
    
    def save_checkpoint(self, path=None):
        path = f"./{self.name}.pt" if path is None else path 

        # Pickle the blocks
        pickled_blocks = pickle.dumps(self.blocks)

        th.save({
            'state_dict': self.state_dict(),
            'pickled_blocks': pickled_blocks,
            'block_network': self.block_network.state_dict(),
            "name": self.name,
            "is_dict_network": self.is_dict_network
        }, path)

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str):
        state = th.load(checkpoint_path)
        model = cls(
            state['name'],
            state['is_dict_network']
        )
        
        # Load the main state dict
        model.load_state_dict(state["state_dict"], strict=False)
        
        # Unpickle and load blocks
        model.blocks = pickle.loads(state['pickled_blocks'])
        for block in model.blocks:
            block.to(model.device)  # Ensure blocks are on the correct device
        
        # Compile the network
        model.compile()
        
        # Load block_network state dict
        model.block_network.load_state_dict(state['block_network'])
        
        return model