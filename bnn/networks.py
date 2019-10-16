import torch.nn as nn

from torch.nn import Conv2d, MaxPool2d, Linear
from torch.nn.functional import relu, hardtanh

from binary_models import (
    BinaryConv2d,
    BinaryLinear,
    MomentumWithThresholdBinaryOptimizer,
)


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class BinaryNet(nn.Module):
    def __init__(self):
        super(BinaryNet, self).__init__()

    pass


class RealVggCloneNet(nn.Module):
    def __init__(self):
        super(RealVggCloneNet, self).__init__()

        self.conv1a = Conv2d(3, 128, 3, padding=1)
        self.conv1b = Conv2d(128, 128, 3, padding=1)

        self.p1 = MaxPool2d(2)

        self.conv2a = Conv2d(128, 256, 3, padding=1)
        self.conv2b = Conv2d(256, 256, 3, padding=1)

        self.p2 = MaxPool2d(2)

        self.conv3a = Conv2d(256, 512, 3, padding=1)
        self.conv3b = Conv2d(512, 512, 3, padding=1)

        self.p3 = MaxPool2d(2)

        self.flatten = Flatten()

        self.fc1 = Linear(512 * 4 * 4, 1024)
        self.fc2 = Linear(1024, 1024)
        # softmax is done by loss function (criterion)

    def forward(self, x):
        x = relu(self.conv1a(x))
        x = relu(self.conv1b(x))
        x = self.p1(x)

        x = relu(self.conv2a(x))
        x = relu(self.conv2b(x))
        x = self.p2(x)

        x = relu(self.conv3a(x))
        x = relu(self.conv3b(x))
        x = self.p3(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.fc2(x)

        return x


################################################################################


class ExampleBinaryNet(nn.Module):
    def __init__(self):
        super(ExampleBinaryNet, self).__init__()

        self.conv1 = BinaryConv2d(3, 100, 5)
        self.conv2 = BinaryConv2d(100, 16, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = BinaryLinear(16 * 5 * 5, 120)
        self.fc2 = BinaryLinear(120, 84)
        self.fc3 = BinaryLinear(84, 10)

    def forward(self, x):
        x = self.pool(hardtanh(self.conv1(x)))
        x = self.pool(hardtanh(self.conv2(x)))

        x = x.view(-1, 16 * 5 * 5)
        x = hardtanh(self.fc1(x))
        x = hardtanh(self.fc2(x))
        x = self.fc3(x)

        return x


class ExampleNet(nn.Module):
    def __init__(self):
        super(ExampleNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 100, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(100, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(relu(self.conv1(x)))
        x = self.pool(relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)

        return x


class TrivialBinaryNetwork(nn.Module):
    def __init__(self, in_features: int = 10):
        super(TrivialBinaryNetwork, self).__init__()

        self.layers = nn.Sequential(nn.Linear(in_features, out_features=2))

    def forward(self, x):
        self.layers.forward(x)
