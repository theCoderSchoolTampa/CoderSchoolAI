from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import torch as th
import torch.nn as nn
from CoderSchoolAI.Environment.Attributes import *
from CoderSchoolAI.Neural.Block import *
import numpy as np

from CoderSchoolAI.Neural.Block import Block


class LinearBlock(Block):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_hidden_layers: int = 3,
        hidden_size: Union[int, List[int]] = 128,
        dropout: Optional[float] = None,
        activation: Optional[Callable] = None,
        use_layer_norm: bool = False,
        device: th.device = th.device("cpu"),
    ):  # TODO: Add LayerNorm option on the End of the LinearBlock.
        """
        Creates a LinearBlock for a Neural Network.
           - input_size (int): The size of the input to the block.
           - output_size (int): The size of the output from the block.
           - num_hidden_layers (int, optional): The number of hidden layers in the block. Default is 3.
           - hidden_size (int or List[int], optional): The size(s) of the hidden layer(s) in the block. If an int is provided, all hidden layers will have the same size. If a List[int] is provided, each element corresponds to the size of a specific hidden layer. Default is 128.
           - dropout (float, optional): The dropout probability for the hidden layers. Default is None, which means no dropout is applied.
           - activation (Callable, optional): The activation function to be used for the hidden layers. Default is None, which means no activation function is applied.
           - device (torch.device, optional): The device on which the computations will be performed. Default is 'cpu'.
        """

        activation_function = activation if activation is not None else nn.ReLU
        super(LinearBlock, self).__init__(
            b_type=Block.Type.LINEAR,
            activation_function=activation_function,
            device=device,
        )
        """save these as self attributes"""
        self.input_size = input_size
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self._use_dropout = dropout is not None and self.dropout > 0.0
        self._use_layer_norm = use_layer_norm
        self._use_custom_hidden = isinstance(hidden_size, list)
        
        if self._use_custom_hidden:
            assert (
                len(hidden_size) == self.num_hidden_layers
            )  # Ensure that hidden_size is a list of the same length as num_hidden_layers

        self.module = self.regenerate_network()
        
        self.to(self.device)

    def _get_join_block(
        self: Block,
    ):
        """
        Will retrieve the shallowest JoinBlock in the network.
        """
        if self.forward_connections is None:
            return None
        if self.forward_connections.d_type == Block.Type.JOIN:
            return self.forward_connections

        return self._get_join_block(self.forward_connections)

    def join_block(self, block: Block, key: str = None):
        """
        Linear Blocks can be joined with other LinearBlocks, JoinBlocks, or OutputBlocks.
        - Throws an error if the input block is not of the types listed above.
        """
        if isinstance(block, Block):
            if block.b_type == Block.Type.LINEAR:
                if not self.output_size == block.input_size:
                    print(
                        f"InputBlock.join_block() expects a LinearBlock with the same input_size as the Output of this block. Rebuilding the Block's Network."
                    )
                    block.regenerate_network()
                self.forward_connections = block
            elif block.b_type == Block.Type.JOIN or block.b_type == Block.Type.OUTPUT:
                self.forward_connections = block
            else:
                raise Exception(
                    f"InputBlock.join_block() expects a LinearBlock, JoinBlock, or OutputBlock. Provided {type(block)}, {block}"
                )
        else:
            raise ValueError("InputBlock.join_block() expects a Block object")

    def forward(self, x) -> th.Tensor:
        x = x.to(self.device)
        if len(x.shape) < 2:  # Ensure that the input is a 2D Matrix
            self.forward(x.unsqueeze(0))
        elif len(x.shape) > 2:  # Ensure that the input is a 2D Matrix
            return self.forward(x.squeeze(0))
        if x.shape[-1] != self.input_size:
            raise Exception(
                f"InputBlock.forward() expects a matrix of shape Batch Size x {self.input_size}, Provided Shape: {x.shape}"
            )
        x = self.module(x)
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
        return self.forward_connections(x)

    def regenerate_network(self):
        """
        This function is used to correct/build a network from the internal state/structure of the block.
        """
        if self._use_custom_hidden:
            layer_size = np.concatenate([self.hidden_size, [self.output_size]])
        else:
            layer_size = np.concatenate(
                [
                    np.full((self.num_hidden_layers,), self.hidden_size),
                    [self.output_size],
                ]
            )
        self.layers = []
        prev_size = self.input_size
        for i in range(len(layer_size)):
            self.layers.append(
                nn.Linear(int(prev_size), int(layer_size[i]), device=self.device)
            )
            self.layers.append(self.activation_function())
            if self._use_dropout:
                self.layers.append(nn.Dropout(p=self.dropout))
            prev_size = layer_size[i]
            
        if len(self.layers) > 0 and self._use_layer_norm:
            self.layers.append(nn.LayerNorm(prev_size))
            
        return nn.Sequential(*self.layers)

    def copy(self):
        """
        Copies a LinearBlock to a new LinearBlock with the same parameters.
        """
        new_copy = LinearBlock(
            self.input_size,
            self.output_size,
            self.num_hidden_layers,
            self.hidden_size,
            self.dropout,
            self.activation_function,
            self.device,
        )
        new_copy.module.load_state_dict(self.module.state_dict())
        return new_copy

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
