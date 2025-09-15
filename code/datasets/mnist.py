"""
:file: mnist.py
:date: 2025-07-10
:description: Loader for MNIST dataset
"""


from torchvision import datasets, transforms
from torchvision.datasets import MNIST


MNIST_NUM_CHANNEL = 1 
MNIST_NUM_CLASS = 10

MNIST_MEAN = (0.1307, )
MNIST_STD = (0.3081, )

MNIST_ROOT = '../../data/mnist/'


def get_dataset(train=True, is_transform=True) -> MNIST:
    """Get MNIST dataset

    :param train: Whether to load training set, defaults to True
    :param is_transform: Whether to apply transformations, defaults to True
    :return: MNIST dataset
    """
    if is_transform:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ])
    else:
        transform = transforms.ToTensor()

    return MNIST(root=MNIST_ROOT, train=train, download=True, transform=transform)


