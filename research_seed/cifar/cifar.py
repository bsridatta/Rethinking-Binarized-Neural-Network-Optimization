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
            BinaryConv2d(3, 128, kernel_size=3,
                         stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.Hardtanh(inplace=True),
            # layer 2
            BinaryConv2d(128, 128, kernel_size=3,
                         stride=1, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.Hardtanh(inplace=True),
            # layer 3
            BinaryConv2d(128, 256, kernel_size=3,
                         stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.Hardtanh(inplace=True),
            # layer 4
            BinaryConv2d(256, 256, kernel_size=3,
                         stride=1, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.Hardtanh(inplace=True),
            # layer 5
            BinaryConv2d(256, 512, kernel_size=3,
                         stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True),
            # layer 6
            BinaryConv2d(512, 512, kernel_size=3,
                         stride=1, padding=1, bias=True),
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
        logger_logs = {'train_acc': train_acc,
                       'train_loss': train_loss
                       }

        # loss is strictly required
        output = OrderedDict({
            "loss": train_loss,
            "progress_bar": {"train_acc": train_acc},
            "log": logger_logs
        })

        return output

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)

        # validation metrics for monitoring
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_loss = F.cross_entropy(y_hat, y)

        output = OrderedDict({
            "val_loss": val_loss,
            "val_acc":  torch.tensor(val_acc)  # must be a tensor
        })

        return output

    def validation_end(self, outputs):
        """
        outputs -- list of outputs ftom each validation step
        """
        # The outputs here are strictly for progress bar
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        logger_logs = {'val_avg_acc': avg_acc,
                       'val_avg_loss': avg_loss
                       }

        output = OrderedDict({
            "progress_bar": logger_logs,
            "log": logger_logs
        })

        return output

    def configure_optimizers(self):
        optimizer = MomentumWithThresholdBinaryOptimizer(
            self.parameters(), ar=self.ar, threshold=self.t
        )

        for param_idx, p in enumerate(self.parameters()):
            optimizer.total_weights[param_idx] = len(p)

        return optimizer

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
        """
        Adaptivity rate decay
        """
        # Decay every 100 epochs
        if current_epoch % 100 == 0:
            self.ar *= 0.1

        # update params - optimizer step
        flips_curr_step = optimizer.step()
        pi = np.asarray([0]*len(optimizer.total_weights.keys()))

        for idx in flips_curr_step.keys() & optimizer.total_weights.keys():
            pi[idx] = flips_curr_step[idx] / \
                optimizer.total_weights[idx] + 10**-9

        ''' log pi from layer 0 to 33 '''
        # self.logger.experiment.add_histogram(
        #     tag="b"+str(batch_nb), values=pi, global_step=self.trainer.global_step)

        optimizer.zero_grad()

    def on_after_backward(self):
        # logging updated weights
        if self.trainer.global_step % 1000 == 0:
            for id, p in enumerate(self.parameters()):
                self.logger.experiment.add_histogram(
                    tag="l"+str(id), values=p, global_step=self.trainer.global_step)

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        train_data = CIFAR10(
            os.getcwd(),
            train=True,
            download=True,
            transform=train_val_transform
        )

        start = 0
        end = 5  # 0000

        data_loader = DataLoader(train_data,
                                 batch_size=self.hparams.batch_size,
                                 sampler=SubsetRandomSampler(
                                     range(start, end)
                                 )
                                 )

        print("train len ", len(data_loader))
        return data_loader

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        val_data = CIFAR10(
            os.getcwd(),
            train=True,
            download=True,
            transform=train_val_transform
        )

        start = 0  # 40000
        end = 5  # 0000  # len(val_data)

        data_loader = DataLoader(val_data,
                                 batch_size=self.hparams.batch_size,
                                 sampler=SubsetRandomSampler(
                                     range(start, end)
                                 )
                                 )

        print("val len ", len(data_loader))
        return data_loader

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(
            CIFAR10(os.getcwd(), train=False, download=True,
                    transform=test_transform),
            batch_size=self.hparams.batch_size,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument("--adaptivity-rate", default=10**-4, type=float)
        parser.add_argument("--threshold", default=10**-8, type=float)
        parser.add_argument("--batch_size", default=50, type=int)

        return parser
