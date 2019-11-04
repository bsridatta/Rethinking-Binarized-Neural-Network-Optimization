"""
This file defines the core research contribution
"""
import os

from collections import OrderedDict

import numpy as np

# import math

import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from argparse import ArgumentParser

import pytorch_lightning as pl

from bytorch import MomentumWithThresholdBinaryOptimizer
from bytorch import BinaryLinear, BinaryConv2d

train_val_transform = transforms.Compose(
    [
        transforms.Pad((4, 4, 4, 4)),
        transforms.RandomCrop((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # values are between [0, 1], we want [-1, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # values are between [0, 1], we want [-1, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

num_classes = 10


class BnnOnCIFAR10(pl.LightningModule):
    def __init__(self, hparams):
        super(BnnOnCIFAR10, self).__init__()

        self.hparams = hparams

        self.ar = hparams.adaptivity_rate
        self.t = hparams.threshold
        self.bs = hparams.batch_size
        self.adam_lr = hparams.adam_lr
        self.decay_n_epochs = hparams.decay_n_epochs
        self.decay_exponential = hparams.decay_exponential

        print(self.decay_n_epochs, self.decay_exponential)

        self.features = nn.Sequential(
            OrderedDict(
                [
                    # layer 1
                    (
                        "binary1",
                        BinaryConv2d(
                            3,
                            128,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                            binarize_input=False,
                        ),
                    ),
                    ("bn1", nn.BatchNorm2d(128)),
                    # layer 2
                    (
                        "binary2",
                        BinaryConv2d(
                            128,
                            128,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                            binarize_input=True,
                        ),
                    ),
                    ("mp2", nn.MaxPool2d(kernel_size=2, stride=2)),
                    ("bn2", nn.BatchNorm2d(128)),
                    # layer 3
                    (
                        "binary3",
                        BinaryConv2d(
                            128,
                            256,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                            binarize_input=True,
                        ),
                    ),
                    ("bn3", nn.BatchNorm2d(256)),
                    # layer 4
                    (
                        "binary4",
                        BinaryConv2d(
                            256,
                            256,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                            binarize_input=True,
                        ),
                    ),
                    ("mp4", nn.MaxPool2d(kernel_size=2, stride=2)),
                    ("bn4", nn.BatchNorm2d(256)),
                    # layer 5
                    (
                        "binary5",
                        BinaryConv2d(
                            256,
                            512,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                            binarize_input=True,
                        ),
                    ),
                    ("bn5", nn.BatchNorm2d(512)),
                    # layer 6
                    (
                        "binary6",
                        BinaryConv2d(
                            512,
                            512,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                            binarize_input=True,
                        ),
                    ),
                    ("mp6", nn.MaxPool2d(kernel_size=2, stride=2)),
                    ("bn6", nn.BatchNorm2d(512)),
                ]
            )
        )

        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    # layer 7
                    (
                        "binary7",
                        BinaryLinear(512 * 4 * 4, 1024, bias=True, binarize_input=True),
                    ),
                    ("bn7", nn.BatchNorm1d(1024)),
                    # layer 8
                    (
                        "binary8",
                        BinaryLinear(1024, 1024, bias=True, binarize_input=True),
                    ),
                    ("bn8", nn.BatchNorm1d(1024)),
                    # layer 9
                    (
                        "binary9",
                        BinaryLinear(1024, num_classes, bias=True, binarize_input=True),
                    ),
                    ("bn9", nn.BatchNorm1d(10)),
                ]
            )
        )

        self._num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self._prev_epoch = -1

    def binary_parameters(self):
        for name, layer in self.named_parameters():
            if "binary" in name:
                yield layer

    def non_binary_parameters(self):
        for name, layer in self.named_parameters():
            if "bn" in name:
                yield layer

    def forward(self, x):
        # REQUIRED
        x = self.features(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)

        # Training metrics for monitoring
        labels_hat = torch.argmax(y_hat, dim=1)
        train_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        train_loss = F.cross_entropy(y_hat, y)
        logger_logs = {"train_acc": train_acc, "train_loss": train_loss}

        # loss is strictly required
        output = OrderedDict(
            {
                "loss": train_loss,
                "progress_bar": {"train_acc": train_acc},
                "log": logger_logs,
            }
        )

        return output

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)

        # validation metrics for monitoring
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_loss = F.cross_entropy(y_hat, y)

        output = OrderedDict(
            {"val_loss": val_loss, "val_acc": torch.tensor(val_acc)}  # must be a tensor
        )

        return output

    def validation_end(self, outputs):
        """
        outputs -- list of outputs ftom each validation step
        """
        # The outputs here are strictly for progress bar
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()

        logger_logs = {"val_acc": avg_acc, "val_loss": avg_loss}

        output = OrderedDict({"progress_bar": logger_logs, "log": logger_logs})

        return output

    def configure_optimizers(self):
        return MomentumWithThresholdBinaryOptimizer(
            self.binary_parameters(),
            self.non_binary_parameters(),
            ar=self.ar,
            threshold=self.t,
            adam_lr=self.adam_lr,
        )

    def optimizer_step(
        self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None
    ):
        """
        Adaptivity rate decay
        """
        if self._prev_epoch is not current_epoch:
            self._prev_epoch = current_epoch
        else:
            return

        # Decay every 100 epochs
        if current_epoch % self.decay_n_epochs == 0 and current_epoch != 0:
            print("reduced :o")
            self.ar *= self.decay_exponential

        # update params - optimizer step
        flips_curr_step = optimizer.step(ar=self.ar)

        sum_flips = sum(flips_curr_step.values())
        pi = np.log(sum_flips / (self._num_params + np.e - 9))

        self.logger.experiment.log(({"pi": pi, "flips": sum_flips, "ar": self.ar}))

        optimizer.zero_grad()

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        train_data = CIFAR10(
            os.getcwd(), train=True, download=True, transform=train_val_transform
        )

        data_loader = DataLoader(train_data, batch_size=self.hparams.batch_size)

        print("train len ", len(data_loader))
        return data_loader

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        val_data = CIFAR10(
            os.getcwd(), train=True, download=True, transform=train_val_transform
        )

        data_loader = DataLoader(val_data, batch_size=self.hparams.batch_size)

        print("val len ", len(data_loader))
        return data_loader

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(
            CIFAR10(os.getcwd(), train=False, download=True, transform=test_transform),
            batch_size=self.hparams.batch_size,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument("--adaptivity-rate", default=10 ** -4, type=float)
        parser.add_argument("--threshold", default=10 ** -8, type=float)
        parser.add_argument("--batch_size", default=50, type=int)
        parser.add_argument("--adam-lr", default=0.01, type=float)
        parser.add_argument("--decay-n-epochs", default=100, type=int)
        parser.add_argument("--decay-exponential", default=0.1, type=float)

        return parser
