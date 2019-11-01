import os

import torch as t
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as opt

from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from torchsummary import summary

from binary_models import (
    MomentumWithThresholdBinaryOptimizer,
    BinaryLinear,
    BinaryConv2d,
)

from matplotlib import pyplot as plt

t.manual_seed(424121)
group_a_generator = dist.Normal(0.8, 0.001)
group_b_generator = dist.Normal(0, 0.001)
group_c_generator = dist.Normal(-0.8, 0.001)


def generate_data(test=False):
    if test:
        return DataLoader(
            MNIST(
                os.getcwd(), train=True, download=True, transform=transforms.ToTensor()
            ),
            batch_size=64,
        )
    else:
        return DataLoader(
            MNIST(
                os.getcwd(), train=True, download=True, transform=transforms.ToTensor()
            ),
            batch_size=64,
        )


class BinaryNet(nn.Module):
    def __init__(self, in_features, out_features):
        super(BinaryNet, self).__init__()

        self.features = nn.Sequential(
            # layer 1
            BinaryConv2d(1, 32, (3, 3), bias=False),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(),
            # layer 2
            BinaryConv2d(32, 64, (3, 3), bias=False),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(64),
            nn.Hardtanh(),
            # layer 3
            BinaryConv2d(64, 64, (3, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.Hardtanh(),
        )

        self.classifier = nn.Sequential(
            # layer 4
            BinaryLinear(64*7*7, 64),
            nn.BatchNorm1d(64),
            nn.Hardtanh(),
            # layer 5
            BinaryLinear(64, 10)
        )
        # self.fc1 = BinaryLinear(50, 25)
        # self.fc2 = BinaryLinear()
        # # self.bn2 = nn.BatchNorm1d(num_features=50)
        #
        # self.fc2 = BinaryLinear(25, out_features)
        # # self.bn3 = nn.BatchNorm1d(num_features=out_features)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


def main():
    use_gpu = False
    use_binary = True

    n_features, n_classes = 2, 3

    train_loaded = generate_data(test=False)
    test_loaded = generate_data(test=True)

    for _ in range(0, 1):

        if use_binary:
            network: nn.Module = BinaryNet(n_features, n_classes)
            loss_fn = f.multi_margin_loss
            optimizer = MomentumWithThresholdBinaryOptimizer(
                params=network.parameters(), ar=1e-5, threshold=1e-3
            )
        else:
            network: nn.Module = RealNet(n_features, n_classes)
            loss_fn = f.cross_entropy
            optimizer = opt.SGD(network.parameters(), 0.001)

        if use_gpu:
            network = network.to("cuda")

        summary(network, (1, 28, 28), device="cpu")

        for epoch in range(0, 6):
            print("epoch", epoch)

            sum_loss = 0
            total_losses = 0
            total_flips = [0] * 6

            for i, data in enumerate(train_loaded, 0):
                batch, labels = data

                if use_gpu:
                    batch = batch.to("cuda")
                    labels = labels.to("cuda")

                optimizer.zero_grad()

                out = network(batch).squeeze()

                loss = loss_fn(out, labels)
                sum_loss += loss.item()
                total_losses += 1

                loss.backward()

                # for p in network.parameters():
                #     print("###################")
                #     print(p)
                #     print(p.grad)
                #     print()

                flips = optimizer.step()

                if use_binary:
                    total_flips = [a + b for a, b in zip(flips, total_flips)]

            print("average loss", sum_loss / total_losses)
            print(total_flips)

            correct = 0
            total = 0
            with t.no_grad():
                for data in train_loaded:
                    images, labels = data

                    outputs = network(images).squeeze()

                    _, predicted = t.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            train_accuracy = 100 * (correct / total)
            print("train accuracy:", train_accuracy)

            # print(total_flips)

        correct = 0
        total = 0
        with t.no_grad():
            for data in train_loaded:
                images, labels = data

                outputs = network(images).squeeze()

                _, predicted = t.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        train_accuracy = 100 * (correct / total)

        correct = 0
        total = 0
        with t.no_grad():
            for data in test_loaded:
                images, labels = data

                outputs = network(images).squeeze()

                _, predicted = t.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * (correct / total)

        print(
            f"train accuracy: {train_accuracy: .3f} test accuracy: {test_accuracy: .3f}"
        )


if __name__ == "__main__":
    main()
