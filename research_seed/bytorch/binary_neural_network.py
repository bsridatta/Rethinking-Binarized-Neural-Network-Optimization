###############################################################################
#
# Provides the required elements to implement a binary convolutional network in
# PyTorch.
#
# This file contains the following elements are implemented:
# * BinaryLinear
# * BinaryConv2d
# * sign function with straight-through estimator gradient
# * Binary optimization algorithm
#
# Inspiration taken from:
# https://github.com/itayhubara/BinaryNet.pytorch/blob/master/models/binarized_modules.py
#
# Author(s): Nik Vaessen
###############################################################################

from typing import TypeVar, Union, Tuple, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as f

from torch import Tensor
from torch.autograd import Function
from torch.optim.optimizer import Optimizer
from torch.optim import Adam

################################################################################

# taken from https://github.com/pytorch/pytorch/blob/bfeff1eb8f90aa1ff7e4f6bafe9945ad409e2d97/torch/nn/common_types.pyi

T = TypeVar("T")
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]
_size_2_t = _scalar_or_tuple_2_t[int]

################################################################################
# Quantizers


class Binarize(Function):
    clip_value = 1

    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)

        output = inp.sign()

        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp: Tensor = ctx.saved_tensors[0]

        clipped = inp.abs() <= Binarize.clip_value

        output = torch.zeros(inp.size()).to(grad_output.device)
        output[clipped] = 1
        output[~clipped] = 0

        return output * grad_output


binarize = Binarize.apply

################################################################################
# Optimizers for binary networks

class MomentumWithThresholdBinaryOptimizer(Optimizer):
    def __init__(
        self,
        binary_params,
        bn_params,
        ar: float = 0.0001,
        threshold: float = 0,
        adam_lr=0.001,
    ):
        if not 0 < ar < 1:
            raise ValueError(
                "given adaptivity rate {} is invalid; should be in (0, 1) (excluding endpoints)".format(
                    ar
                )
            )
        if threshold < 0:
            raise ValueError(
                "given threshold {} is invalid; should be > 0".format(threshold)
            )

        self.total_weights = {}
        self._adam = Adam(bn_params, lr=adam_lr)

        defaults = dict(adaptivity_rate=ar, threshold=threshold)
        super(MomentumWithThresholdBinaryOptimizer, self).__init__(
            binary_params, defaults
        )

    def step(self, closure: Optional[Callable[[], float]] = ..., ar=None):
        self._adam.step()

        flips = {None}

        for group in self.param_groups:
            params = group["params"]

            y = group["adaptivity_rate"]
            t = group["threshold"]
            flips = {}

            if ar is not None:
                y = ar

            for param_idx, p in enumerate(params):
                grad = p.grad.data
                state = self.state[p]

                if "moving_average" not in state:
                    m = state["moving_average"] = torch.clone(grad).detach()
                else:
                    m: Tensor = state["moving_average"]

                    m.mul_((1 - y))
                    m.add_(grad.mul(y))

                mask = (m.abs() >= t) * (m.sign() == p.sign())
                mask = mask.double() * -1
                mask[mask == 0] = 1

                flips[param_idx] = (mask == -1).sum().item()

                p.data.mul_(mask)

        return flips

    def zero_grad(self) -> None:
        super().zero_grad()
        self._adam.zero_grad()


################################################################################
# binary torch layers


class BinaryLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias=False,
        keep_latent_weight=False,
        binarize_input=False,
    ):
        super().__init__(in_features, out_features, bias=bias)

        self.keep_latent_weight = keep_latent_weight
        self.binarize_input = binarize_input

        if not self.keep_latent_weight:
            with torch.no_grad():
                self.weight.data.sign_()
                self.bias.data.sign_() if self.bias is not None else None

    def forward(self, inp: Tensor) -> Tensor:
        if self.keep_latent_weight:
            weight = binarize(self.weight)
        else:
            weight = self.weight

        bias = self.bias if self.bias is None else binarize(self.bias)

        if self.binarize_input:
            inp = binarize(inp)

        return f.linear(inp, weight, bias)


class BinaryConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride=1,
        padding=1,
        bias=False,
        keep_latent_weight=False,
        binarize_input=False,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )

        self.keep_latent_weight = keep_latent_weight
        self.binarize_input = binarize_input

        if not self.keep_latent_weight:
            with torch.no_grad():
                self.weight.data.sign_()
                self.bias.data.sign_() if self.bias is not None else None

    def forward(self, inp: Tensor) -> Tensor:
        if self.keep_latent_weight:
            weight = binarize(self.weight)
        else:
            weight = self.weight

        bias = self.bias if self.bias is None else binarize(self.bias)

        if self.binarize_input:
            inp = binarize(inp)

        return f.conv2d(
            inp, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )
