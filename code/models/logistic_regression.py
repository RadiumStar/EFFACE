"""
File: logistic_regression.py
Date: 2025-07-10
Description: Logistic Regression models
"""


import torch 
import torch.nn as nn 


class LogisticRegression(nn.Module): 
    def __init__(self, input_channel=1, input_dim=784, num_class=10):
        """logostic regression model

        :param input_channels: channel of image, defaults to 1
        :param input_dim: dimension of image, defaults to 784
        :param num_classes: number of class, defaults to 10
        """
        super().__init__()
        self.input_channel = input_channel
        self.input_dim = input_dim
        self.num_class = num_class
        self.linear = nn.Linear(input_channel * input_dim, num_class, bias=False)

    def forward(self, x):
        """forward function

        :param x: input data
        :return: output of model
        """
        x = x.view(x.size(0), -1)
        return self.linear(x).squeeze(1) if self.num_class == 1 else self.linear(x)
    

def LogisticRegression_BINARY_MNIST(input_channel=1, input_dim=784, num_class=1):
    """Logistic Regression model for MNIST dataset

    :param input_channel: channel of image, defaults to 1
    :param input_dim: dimension of image, defaults to 784
    :param num_class: number of class, defaults to 10
    :return: LogisticRegression model
    """
    return LogisticRegression(input_channel=input_channel, input_dim=input_dim, num_class=num_class)


def LogisticRegression_MNIST(input_channel=1, input_dim=784, num_class=10):
    """Logistic Regression model for MNIST dataset

    :param input_channel: channel of image, defaults to 1
    :param input_dim: dimension of image, defaults to 784
    :param num_class: number of class, defaults to 10
    :return: LogisticRegression model
    """
    return LogisticRegression(input_channel=input_channel, input_dim=input_dim, num_class=num_class)


if __name__ == "__main__":
    # Example usage
    model = LogisticRegression_BINARY_MNIST()
    print(model)
    x = torch.randn(32, 1, 28, 28)  # Batch of 32 MNIST images
    output = model(x)
    print("Output shape:", output.shape)  # Should be (32, 10) for MNIST