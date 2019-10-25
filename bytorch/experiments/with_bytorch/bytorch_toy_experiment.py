import torch as t
import torch.nn as nn
import torch.nn.functional as f
import torch.distributions as dist
import torch.optim as opt
import torch.utils.data as dutils

from binary_models import MomentumWithThresholdBinaryOptimizer, BinaryLinear

# t.manual_seed(424121)
group_a_generator = dist.Normal(0.8, 0.001)
group_b_generator = dist.Normal(0, 0.001)
group_c_generator = dist.Normal(0.8, 0.001)


class ToyDataset(dutils.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


def get_one_hot(hot_index, total_classes):
    if 0 <= hot_index < total_classes:
        empty = t.zeros(total_classes)
        empty[hot_index] = 1

        return empty

    raise ValueError("cannot go outside range of {}".format(total_classes))


def generate_data(n_samples=1024, n_features=100):
    a, b, c = [], [], []

    for _ in range(0, n_samples):
        a.append((group_a_generator.sample((1, n_features)), 0))
        b.append((group_b_generator.sample((1, n_features)), 1))
        c.append((group_c_generator.sample((1, n_features)), 2))

    return ToyDataset([] + a + b + c)


class RealNet(nn.Module):
    def __init__(self, in_features, out_features):
        super(RealNet, self).__init__()

        self.fc1 = nn.Linear(in_features, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, out_features)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class BinaryNet(nn.Module):
    def __init__(self, in_features, out_features):
        super(BinaryNet, self).__init__()

        self.fc1 = BinaryLinear(in_features, 50)
        # self.bn1 = nn.BatchNorm1d(num_features=100)

        self.fc2 = BinaryLinear(50, 25)
        # self.bn2 = nn.BatchNorm1d(num_features=50)

        self.fc3 = BinaryLinear(25, out_features)
        # self.bn3 = nn.BatchNorm1d(num_features=out_features)

    def forward(self, x):
        # x = f.hardtanh(self.bn1(self.fc1(x)))
        # x = f.hardtanh(self.bn2(self.fc2(x)))
        # x = self.bn3(self.fc3(x))

        x = f.hardtanh(self.fc1(x))
        x = f.hardtanh(self.fc2(x))
        x = self.fc3(x)

        return x


def main():
    use_gpu = False
    use_binary = True

    n_features, n_classes = 100, 3

    train = generate_data(n_samples=1024, n_features=n_features)
    test = generate_data(n_samples=100, n_features=n_features)

    train_loaded = dutils.DataLoader(train, batch_size=16)
    test_loaded = dutils.DataLoader(test, batch_size=16)

    network: nn.Module = BinaryNet(n_features, n_classes)

    if use_gpu:
        network = network.to("cuda")

    if use_binary:
        network: nn.Module = BinaryNet(n_features, n_classes)
        loss_fn = f.multi_margin_loss
        optimizer = MomentumWithThresholdBinaryOptimizer(
            params=network.parameters(), ar=1e-3, threshold=1e-3
        )
    else:
        network: nn.Module = RealNet(n_features, n_classes)
        loss_fn = f.cross_entropy
        optimizer = opt.SGD(network.parameters(), 0.001)

    for epoch in range(0, 100):
        print("epoch", epoch, end=" ")
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
            flips = optimizer.step()
            # print(flips)
            total_flips = [a + b for a, b in zip(flips, total_flips)]

        print(sum_loss / total_losses, end=" ")
        print(total_flips)

    correct = 0
    total = 0
    with t.no_grad():
        for data in test_loaded:
            images, labels = data

            outputs = network(images).squeeze()

            _, predicted = t.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        "Accuracy of the network on the test samples: %d %%" % (100 * correct / total)
    )


if __name__ == "__main__":
    main()
