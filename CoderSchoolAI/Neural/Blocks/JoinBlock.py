from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import torch as th
import torch.nn as nn
from CoderSchoolAI.Environment.Attributes import *
from CoderSchoolAI.Neural.Block import *
import numpy as np
from enum import Enum


from CoderSchoolAI.Neural.Block import Block


class JoinBlock(Block):
    def __init__(
        self,
        join_size: int,
        device: th.device = th.device("cpu"),
        activation: Optional[Callable] = None,
    ):
        """
        Creates a Join for a Dictionary Neural Network.
        - join_size (int): The size of the join (aka. the sum of output sizes)
        - activation (Callable, optional): The activation function to be used for the hidden layers. Default is None, which means no activation function is applied.
        - device (torch.device, optional): The device on which the computations will be performed. Default is 'cpu'.
        """
        super(JoinBlock, self).__init__(
            b_type=Block.Type.JOIN, activation_function=activation, device=device
        )
        self.join_size = join_size

    def join_block(self, block: "Block"):
        """
        Joins blocks from this block to another block.
        """
        self.forward_connections = block

    def forward(self, x: th.Tensor):
        """
        Forward pass through the block.
        """
        if self.forward_connections is None:
            raise Exception("No forward connections.")
        return self.forward_connections(x)  # TODO: Improve this logic.

    def regenerate_network(self):
        """
        This function is used to correct a network and regenerate it such that it is implemented correctly.
        """
        raise NotImplementedError("This method must be implemented in the Child Block")

    def copy(self):
        return JoinBlock(device=self.device)

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
