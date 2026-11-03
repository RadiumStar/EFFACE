"""
:file: fashion_mnist.py
:date: 2025-07-28
:description: Loader for Fashion MNIST dataset
"""


from torchvision import datasets, transforms
from torchvision.datasets import FashionMNIST


FASHION_MNIST_NUM_CHANNEL = 1
FASHION_MNIST_NUM_CLASS = 10

FASHION_MNIST_MEAN = (0.2860, )
FASHION_MNIST_STD = (0.3530, )

FASHION_MNIST_ROOT = '../../data/fashion_mnist/'


def get_dataset(train=True, is_transform=True) -> FashionMNIST:
    """Get Fashion MNIST dataset

    :param train: Whether to load training set, defaults to True
    :param is_transform: Whether to apply transformations, defaults to True
    :return: Fashion MNIST dataset
    """
    if is_transform:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD),
        ])
    else:
        transform = transforms.ToTensor()

    return FashionMNIST(root=FASHION_MNIST_ROOT, train=train, download=True, transform=transform)