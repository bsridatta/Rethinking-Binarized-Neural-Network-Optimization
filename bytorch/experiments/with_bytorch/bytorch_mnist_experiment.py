import os
from collections import OrderedDict

import torch as t
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data.dataloader import DataLoader
from torchsummary import summary
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from binary_models import (
    MomentumWithThresholdBinaryOptimizer,
    BinaryLinear,
    BinaryConv2d,
)

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
            OrderedDict(
                [
                    # layer 1
                    ("binary1", BinaryConv2d(1, 32, (3, 3), bias=False)),
                    ("mp1", nn.MaxPool2d((2, 2))),
                    ("bn1", nn.BatchNorm2d(32)),
                    # ("activation1", nn.Hardtanh()),
                    # layer 2
                    (
                        "binary2",
                        BinaryConv2d(32, 64, (3, 3), bias=False, binarize_input=True),
                    ),
                    ("mp2", nn.MaxPool2d((2, 2))),
                    ("bn2", nn.BatchNorm2d(64)),
                    # ("activation2", nn.Hardtanh()),
                    # layer 3
                    (
                        "binary3",
                        BinaryConv2d(64, 64, (3, 3), bias=False, binarize_input=True),
                    ),
                    ("bn3", nn.BatchNorm2d(64)),
                    # ("activation3", nn.Hardtanh()),
                ]
            )
        )

        self.classifier = nn.Sequential(
            OrderedDict(
                [  # layer 4
                    ("binary4", BinaryLinear(64 * 7 * 7, 64, binarize_input=True)),
                    ("bn4", nn.BatchNorm1d(64)),
                    # ("activation4", nn.Hardtanh()),
                    # layer 5
                    ("binary5", BinaryLinear(64, 10, binarize_input=True)),
                    ("bn5", nn.BatchNorm1d(10)),
                ]
            )
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def binary_parameters(self):
        for name, layer in self.named_parameters():
            if "binary" in name:
                yield layer

    def non_binary_parameters(self):
        for name, layer in self.named_parameters():
            if "bn" in name:
                yield layer


def print_params(network):
    print("\nprinting parameters\n")

    for name, p in network.named_parameters():
        print(name)
        print(p)
        print("gradient")
        print(p.grad)
        print("\n")

    print("################")


def main():
    use_gpu = False

    n_features, n_classes = 2, 3

    train_loaded = generate_data(test=False)
    test_loaded = generate_data(test=True)

    for trail_number in range(0, 100):
        print(f"starting trial {trail_number}")
        network: BinaryNet = BinaryNet(n_features, n_classes)
        # loss_fn = f.multi_margin_loss
        loss_fn = f.cross_entropy

        ar = t.FloatTensor(1, 1).uniform_(1, 10).item()
        tr = t.FloatTensor(1, 1).uniform_(1, 10).item()

        ar = 1e-3
        tr = 1e-5

        print(f"ar={ar}, tr={tr}")

        optimizer = MomentumWithThresholdBinaryOptimizer(
            network.binary_parameters(),
            network.non_binary_parameters(),
            ar=ar,
            threshold=tr,
        )

        if use_gpu:
            network = network.to("cuda")

        # summary(network, (1, 28, 28), device="cpu")

        # print_params(network)

        for epoch in range(0, 6):
            print("epoch", epoch)

            sum_loss = 0
            total_losses = 0
            total_flips = [0 for _ in network.binary_parameters()]

            prev_total_flips = sum(total_flips)

            for i, data in enumerate(train_loaded, 0):
                batch, labels = data
                print(f"\r{i}/{60000/64} - {total_flips}", end="")

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

                total_flips = [a + b for a, b in zip(flips.values(), total_flips)]

                total_flips_current = sum(total_flips)
                # if total_flips_current == prev_total_flips:
                #     break
                # else:
                #     prev_total_flips = total_flips_current
                # if i == 0:
                #     print_params(network)
                # print("step", i, "\n\n")

            print()
            print("average loss", sum_loss / total_losses)

            # print_params(network)


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
