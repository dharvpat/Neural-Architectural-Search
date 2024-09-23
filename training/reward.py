import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from models.nas_model import create_model


def evaluate_model(architecture, epochs=1):
    """
    Train the model for a few epochs and return the reward (e.g., accuracy or negative loss).
    
    Args:
        architecture (list of tuples): The proposed architecture configuration.
        epochs (int): Number of training epochs (1 by default for quick evaluation).
    
    Returns:
        float: The reward, typically accuracy or negative loss.
    """
    # Create model
    model = create_model(architecture)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # CIFAR-10 dataset loading
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Train the model for a few epochs
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    return -running_loss / len(train_loader)  # Return negative loss as reward


def get_complexity(architecture):
    """
    Compute the complexity of the architecture in terms of the number of parameters.
    
    Args:
        architecture (list of tuples): The proposed architecture configuration.
    
    Returns:
        int: Number of parameters in the model.
    """
    model = create_model(architecture)
    return sum(p.numel() for p in model.parameters())


def calculate_reward(architecture, alpha=0.01):
    """
    Calculate the reward for an architecture based on performance and complexity.

    Args:
        architecture (list of tuples): The architecture configuration.
        alpha (float): The complexity penalty factor (default=0.01).

    Returns:
        float: The reward (higher is better).
    """
    # Get reward based on training performance
    performance_reward = evaluate_model(architecture)

    # Penalize large models (number of parameters)
    complexity_penalty = alpha * get_complexity(architecture)

    return performance_reward - complexity_penalty