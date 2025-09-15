"""
:file: binary_mnist.py
:date: 2025-07-11
:description: mnist dataset for binary classification(3 vs 8)
"""

from typing import Tuple 

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from PIL import Image


BINARY_MNIST_NUM_CHANNEL = 1 
BINARY_MNIST_NUM_CLASS = 2

MNIST_MEAN = (0.1307, )
MNIST_STD = (0.3081, )

MNIST_ROOT = '../../data/mnist/'


class BinaryMNIST(Dataset): 
    def __init__(self, root: str=MNIST_ROOT, binary_class: Tuple[int, int]=(3, 8), train=True, transform=None):
        """Binary MNIST dataset for binary classification

        :param binary_class: binary digit to classify, defaults to (3, 8)
        :param train: train dataset or test dataset, defaults to True
        :param transform: transformations to apply to the data
        """
        self.binary_class = binary_class
        self.train = train
        self.transform = transform

        original_mnist = MNIST(root=root, train=train, download=True, transform=None)
        data, targets = original_mnist.data, original_mnist.targets

        mask = (targets == binary_class[0]) | (targets == binary_class[1])
        data = data[mask]
        targets = targets[mask]

        # Map binary classes to 0 and 1
        targets = (targets == binary_class[1]).long()

        self.data = data
        self.targets = targets.to(torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.targets[idx]

        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform:
            img = self.transform(img)

        return img, label


def get_dataset(train=True, is_transform=True) -> Dataset:
    """Get Binary MNIST dataset

    :param train: Whether to load training set, defaults to True
    :param is_transform: Whether to apply transformations, defaults to True
    :return: Binary MNIST dataset
    """
    if is_transform:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ])
    else:
        transform = transforms.ToTensor()

    return BinaryMNIST(root=MNIST_ROOT, train=train, transform=transform)


if __name__ == "__main__":
    dataset = get_dataset(train=True, is_transform=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for images, labels in dataloader:
        print("Images shape:", images.shape)  # should be (batch_size, 1, 28, 28)
        print("Labels shape:", labels.shape)  # should be (batch_size,)
        print("Labels:", labels)
        break