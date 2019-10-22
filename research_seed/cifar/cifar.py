"""
This file defines the core research contribution   
"""
import os

import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler


from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from argparse import ArgumentParser

import pytorch_lightning as pl

from bnn import MomentumWithThresholdBinaryOptimizer
from bnn import BinaryLinear, BinaryConv2d

train_val_transform = transforms.Compose(
    [
        transforms.Pad((4, 4, 4, 4)),
        transforms.RandomCrop((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # values are between [0, 1], we want [-1, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # values are between [0, 1], we want [-1, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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

        self.features = nn.Sequential(
            # layer 1
            BinaryConv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.Hardtanh(inplace=True),
            # layer 2
            BinaryConv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.Hardtanh(inplace=True),
            # layer 3
            BinaryConv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.Hardtanh(inplace=True),
            # layer 4
            BinaryConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.Hardtanh(inplace=True),
            # layer 5
            BinaryConv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True),
            # layer 6
            BinaryConv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True),
        )

        self.classifier = nn.Sequential(
            # layer 1
            BinaryLinear(512 * 4 * 4, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            # layer 2
            BinaryLinear(1024, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            # layer 3
            BinaryLinear(1024, num_classes, bias=True),
            nn.BatchNorm1d(num_classes, affine=False),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.classifier(x)

        return x

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch

        y_hat = self.forward(x)

        return {"loss": F.cross_entropy(y_hat, y)}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {"val_loss": F.cross_entropy(y_hat, y)}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"avg_val_loss": avg_loss}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        return MomentumWithThresholdBinaryOptimizer(
            self.parameters(), ar=self.ar, threshold=self.t
        )

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED

        return DataLoader(
            CIFAR10(
                os.getcwd(), train=True, download=True, transform=train_val_transform
            ),
            batch_size=self.hparams.batch_size,
            sampler=SubsetRandomSampler([0, 1000])
        )

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(
            CIFAR10(
                os.getcwd(), train=True, download=True, transform=train_val_transform
            ),
            batch_size=self.hparams.batch_size,
            sampler=SubsetRandomSampler([0, 1000])
        )

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(
            CIFAR10(os.getcwd(), train=True, download=True, transform=test_transform),
            batch_size=self.hparams.batch_size,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument("--adaptivity-rate", default=0.01, type=float)
        parser.add_argument("--threshold", default=0.01, type=float)
        parser.add_argument("--batch_size", default=32, type=int)

        return parser
