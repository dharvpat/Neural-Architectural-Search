import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SearchSpace(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SearchSpace, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

    def generate(self, architecture):
        layers = []
        in_channels = self.input_channels

        for out_channels, kernel_size, stride in architecture:
            layers.append(BasicBlock(in_channels, out_channels, kernel_size, stride))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # After convolution layers, you need to calculate the flattened size
        self.flatten = nn.Flatten()

        # Assuming CIFAR-10 input (32x32) with conv layers, you must calculate output size manually or by passing a dummy tensor
        self.dummy_input = torch.randn(1, self.input_channels, 32, 32)  # Example input
        conv_output = self.conv_layers(self.dummy_input)
        flattened_size = conv_output.numel()  # Compute the number of features

        # Add a fully connected layer
        self.fc = nn.Linear(flattened_size, self.num_classes)

        return self

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x