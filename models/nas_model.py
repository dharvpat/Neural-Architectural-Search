import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """
    A simple convolutional block that consists of a convolutional layer,
    batch normalization, and ReLU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


def create_model(architecture):
    """
    Create a neural network model based on the given architecture configuration.

    Args:
        architecture (list of tuples): Each tuple defines (in_channels, out_channels, kernel_size)

    Returns:
        torch.nn.Module: The neural network model.
    """
    layers = []
    in_channels = architecture[0][0]  # The first layer's in_channels

    for (in_channels, out_channels, kernel_size) in architecture:
        layers.append(BasicBlock(in_channels, out_channels, kernel_size))

    # Add a flattening layer and a fully connected output layer (for CIFAR-10, num_classes=10)
    model = nn.Sequential(
        *layers,
        nn.Flatten(),
        nn.Linear(architecture[-1][1] * 32 * 32 // (4 ** len(architecture)), 10)  # Assuming CIFAR-10 (32x32)
    )

    return model