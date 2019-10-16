import torch as t
import torch.nn as nn
import torch.nn.functional as f
import torch.distributions as dist
import torch.optim as opt
import torch.utils.data as dutils

group_a_generator = dist.Normal(10, 4)
group_b_generator = dist.Normal(-10, 4)
group_c_generator = dist.Normal(0, 4)


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


def generate_data(n_samples=100, n_features=100):
    a, b, c = [], [], []

    for _ in range(0, n_samples):
        a.append((group_a_generator.sample((1, n_features)), get_one_hot(0, 3)))
        b.append((group_b_generator.sample((1, n_features)), get_one_hot(1, 3)))
        c.append((group_c_generator.sample((1, n_features)), get_one_hot(2, 3)))

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
        x = f.softmax(self.fc3(x), 2)

        return x


def main():
    n_features, n_classes = 100, 3

    train = generate_data(n_samples=1000, n_features=100)
    test = generate_data(n_samples=100, n_features=100)

    train_loaded = dutils.DataLoader(train, batch_size=4)

    network: nn.Module = RealNet(n_features, n_classes)

    loss = f.cross_entropy
    optimizer = opt.SGD

    for i, data in enumerate(train_loaded, 0):
        batch, labels = data

        out = network.forward(batch)
        print(out)

        break


if __name__ == "__main__":
    main()
