import os

from torch.utils.data import DataLoader


from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


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


def main():
    data_loader = DataLoader(
        CIFAR10(os.getcwd(), train=True, download=True, transform=train_val_transform),
        batch_size=1,
    )

    for batch in data_loader:
        data, label = batch
        print(data)
        print(data.size())
        print(label)

        break


if __name__ == "__main__":
    main()
