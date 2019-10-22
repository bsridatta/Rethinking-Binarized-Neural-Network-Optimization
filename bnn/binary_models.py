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

from typing import TypeVar, Union, Tuple, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch import Tensor
from torch.optim.optimizer import Optimizer

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
# Optimizers for binary networks


class MomentumWithThresholdBinaryOptimizer(Optimizer):
    def __init__(self, params, ar: float = 0.999, threshold: float = 1):
        if not 0 < ar < 1:
            raise ValueError(
                "given adaptivity rate {} is invalid; should be in (0, 1) (excluding endpoints)".format(
                    ar
                )
            )

        if threshold <= 0:
            raise ValueError(
                "given threshold {} is invalid; should be > 0".format(threshold)
            )

        defaults = dict(adaptivity_rate=ar, threshold=threshold)
        super(MomentumWithThresholdBinaryOptimizer, self).__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = ...) -> None:
        for group in self.param_groups:
            params = group["params"]

            y = group["adaptivity_rate"]
            t = group["threshold"]

            for p in params:
                grad = p.grad.data
                state = self.state[p]

                if "moving_average" not in state:
                    m = state["moving_average"] = torch.clone(grad).detach()
                else:
                    m: Tensor = state["moving_average"]

                    m.mul_(1 - y)
                    m.add_(grad.mul(y))

                mask = (m.abs() >= t) * (m.sign() == p.sign())
                mask = mask.double() * -1
                mask[mask == 0] = 1

                p.data.mul_(mask)


class LatentWeightBinaryOptimizer:
    pass


################################################################################
# torch modules


class BinaryLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, latent_weight=False):
        super().__init__(in_features, out_features)

        self.latent_weight = latent_weight

        if not self.latent_weight:
            self.weight.data.sign_()
            self.bias.data.sign_() if self.bias is not None else None

    def forward(self, inp: Tensor) -> Tensor:
        if self.latent_weight:
            weight = to_binary(self.weight)
        else:
            weight = self.weight

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
