from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import torch as th
import torch.nn as nn
from CoderSchoolAI.Environment.Attributes import *
from CoderSchoolAI.Neural.Block import *
import numpy as np

from CoderSchoolAI.Neural.Block import Block


class FlattenBlock(Block):
    def __init__(
        self,
        output_size: int,
        device: th.device = th.device("cpu"),
    ):
        """
        This is the basis for all Deep Neural Network Blocks. This block connects all input data to a corresponding block.
        - output_size: Size of the output tensor on the flatten dimension. You can most likely get away with passing in np.prod(in_attr.shape)
        """
        super(FlattenBlock, self).__init__(
            b_type=Block.Type.FLATTEN, activation_function=None, device=device
        )
        self.output_size = output_size

    def join_block(self, block: Block, key: str = None):
        if key is None:
            self.forward_connections = block
        else:
            self.forward_connections[key] = block

    def forward(self, x) -> th.Tensor:
        x = x.view(x.size(0), self.output_size)
        if (
            self.forward_connections is None
            or (
                isinstance(self.forward_connections, dict)
                and len(self.forward_connections) == 0
            )
            or self.forward_connections.d_type == Block.Type.JOIN
        ):
            return x
        # Another Linear Block or a Output Block
        assert False, "TODO: Update Support for Dict Blocks."
        return self.forward_connections.forward(x)

    def regenerate_network(self):
        """
        This function is used to correct a network and regenerate it such that it is implemented correctly.
        """
        pass

    def copy(self):
        return FlattenBlock(self.output_size, self.device)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        res = f"{type(self).__module__}.{type(self).__qualname__}>,\n"
        return res
