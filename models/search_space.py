import torch.nn as nn

class BasicBlock(nn.Module):
    """
    A basic convolutional block consisting of Conv2D, BatchNorm2D, and ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

def create_model(architecture, num_classes=10):
    """
    Create a neural network model based on the architecture configuration.

    Args:
        architecture (list of tuples): Each tuple contains (in_channels, out_channels, kernel_size, stride)
        num_classes (int): Number of output classes (e.g., 10 for CIFAR-10)

    Returns:
        nn.Sequential: A sequential model built from the architecture.
    """
    layers = []
    
    # CIFAR-10 images have 3 input channels (RGB), the first layer needs to respect this
    in_channels = 3

    for (in_ch, out_channels, kernel_size, stride) in architecture:
        layers.append(BasicBlock(in_channels, out_channels, kernel_size, stride))
        in_channels = out_channels  # Update the in_channels for the next layer

    # Add a global average pooling layer to reduce the feature map size before the final FC layer
    layers.append(nn.AdaptiveAvgPool2d((1, 1)))

    # Flatten the feature map before passing it to the fully connected layer
    layers.append(nn.Flatten())

    # Final fully connected layer, outputting to `num_classes` (e.g., 10 for CIFAR-10)
    layers.append(nn.Linear(in_channels, num_classes))

    return nn.Sequential(*layers)
