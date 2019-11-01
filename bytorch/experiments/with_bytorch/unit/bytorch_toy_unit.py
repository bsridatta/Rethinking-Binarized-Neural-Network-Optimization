import torch as t
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as opt
import torch.utils.data as dutils

from binary_models import MomentumWithThresholdBinaryOptimizer, BinaryLinear, binarize

from matplotlib import pyplot as plt

t.manual_seed(424121)


class BinaryNet(nn.Module):
    def __init__(self, in_features, out_features):
        super(BinaryNet, self).__init__()

        self.fc1 = BinaryLinear(in_features, 3)
        self.fc1.weight.data = t.tensor(
            [[1.0, 1.0, -1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]]
        )

        # self.bn1 = nn.BatchNorm1d(num_features=100)

        self.fc2 = BinaryLinear(3, 3, binarize_input=True)
        self.fc2.weight.data = t.tensor(
            [[1.0, 1.0, -1.0], [-1.0, 1.0, 1.0], [-1.0, 1.0, -1.0]]
        )

        # self.bn2 = nn.BatchNorm1d(num_features=50)

        self.fc3 = BinaryLinear(3, out_features, binarize_input=True)
        self.fc3.weight.data = t.tensor(
            [[-1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]]
        )

        # self.bn3 = nn.BatchNorm1d(num_features=out_features)

    def forward(self, x):
        # x = f.hardtanh(self.bn1(self.fc1(x)))
        # x = f.hardtanh(self.bn2(self.fc2(x)))
        # x = self.bn3(self.fc3(x))

        x1 = self.fc1(x)
        a1 = f.hardtanh(x1)
        print("fc1:", x1)
        print("fc1 act:", a1)

        x2 = self.fc2(a1)
        a2 = f.hardtanh(x2)

        print("fc2:", x2)
        print("fc2 act:", a2)

        y = self.fc3(a2)
        print("fc3:", y)

        return y


def main():
    network: nn.Module = BinaryNet(3, 3)

    print("initial parameters:")
    [print(p) for p in network.parameters()]
    print()

    loss_fn = f.multi_margin_loss
    optimizer = MomentumWithThresholdBinaryOptimizer(
        params=network.parameters(), ar=1e-9, threshold=1e-9
    )

    print("input")
    batch = t.Tensor([0.1, 0.3, -0.5])
    labels = t.Tensor([0]).long()
    print("x=", batch, "y=", labels, end="\n\n")

    losses = []
    for _ in range(0, 1):
        optimizer.zero_grad()

        print("forward step:")
        out = network(batch).squeeze()
        print(out, end="\n\n")

        loss = loss_fn(out, labels)
        losses.append(loss.item())

        loss.backward()

        print("gradient:")
        [print(name, p.grad, sep="\n") for name, p in network.named_parameters()]
        print()

        print("optimization step")
        flips = optimizer.step()
        print()

        print("results:")
        print("loss:", loss)
        print(flips)

        print("new weights:")
        print("initial parameters:")
        [print(p) for p in network.parameters()]
        print()

    for l in losses:
        print(l)


def gradcheck():
    for n in [-10, -2, -1, -0.0001, 0, 0.0001, 1, 2, 10]:
        x = t.tensor(float(n), requires_grad=True)
        y = binarize(x)

        y.backward()

        print(x, y, x.grad)


def loss_check():
    x = t.tensor([1.0, 1.0, 1.0, 1])
    y = t.tensor([0])

    l = f.multi_margin_loss(x, y)
    print(l)


if __name__ == "__main__":
    # gradcheck()
    main()
    # loss_check()
