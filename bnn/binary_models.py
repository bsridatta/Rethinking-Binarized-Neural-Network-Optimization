################################################################################
#
# Provides Pytorch modules for a binary convolution network:
# * BinaryLinear
# * BinaryConv2d
#
# Inspiration taken from:
# https://github.com/itayhubara/BinaryNet.pytorch/blob/master/models/binarized_modules.py
#
# Author(s): Nik Vaessen
################################################################################

import torch.nn as nn
import torch.nn.functional as f

from torch import Tensor

from typing import TypeVar, Union, Tuple

################################################################################

# taken from https://github.com/pytorch/pytorch/blob/bfeff1eb8f90aa1ff7e4f6bafe9945ad409e2d97/torch/nn/common_types.pyi

T = TypeVar("T")
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]
_size_2_t = _scalar_or_tuple_2_t[int]

################################################################################
# Helper functions


def to_binary(inp: Tensor):
    return inp.sign()


################################################################################
# torch modules


class BinaryLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)

    def forward(self, inp: Tensor) -> Tensor:
        weight = to_binary(self.weight)
        bias = self.bias if self.bias is None else to_binary(self.bias)
        inp = to_binary(inp)

        return f.linear(inp, weight, bias)


class BinaryConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t):
        super().__init__(in_channels, out_channels, kernel_size)

    def forward(self, inp: Tensor) -> Tensor:
        weight = to_binary(self.weight)
        bias = self.bias if self.bias is None else to_binary(self.bias)
        inp = to_binary(inp)

        return f.conv2d(
            inp, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

################################################################################
