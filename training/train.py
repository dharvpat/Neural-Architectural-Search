import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from models.search_space import create_model

def train_and_evaluate(architecture):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(architecture)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model.to(device)
    model.train()
    for epoch in range(2):  # Quick evaluation for 1 epoch
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Simple reward: negative loss
    return -loss.item()