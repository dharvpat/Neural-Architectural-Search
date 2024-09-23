import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# LSTM-based controller for NAS
class NASController(nn.Module):
    def __init__(self, num_layers, num_operations, hidden_size):
        super(NASController, self).__init__()
        self.num_operations = num_operations  # This is the number of possible operations (e.g., conv layers)
        self.hidden_size = hidden_size
        
        # LSTM expects input of size `num_operations`
        self.lstm = nn.LSTMCell(input_size=self.num_operations, hidden_size=self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.num_operations)
        self.num_layers = num_layers

    def forward(self, prev_output):
        batch_size = prev_output.size(0)
        
        # Initialize hidden states for LSTM (hx and cx)
        hx, cx = torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size)
        outputs = []
        
        # Iterating over the number of layers to generate each layer's architecture
        for _ in range(self.num_layers):
            hx, cx = self.lstm(prev_output, (hx, cx))  # Pass the previous output through the LSTM
            output = self.fc(hx)  # Fully connected layer to map LSTM output to possible operations
            outputs.append(output)

            # Sample from the logits of the current layer's operations
            prev_output = Categorical(logits=output).sample().unsqueeze(0)  # Add batch dimension if needed

        return outputs

# Training the controller using REINFORCE
def train_controller(controller, architecture, optimizer, reward):
    optimizer.zero_grad()
    log_probs = torch.stack([Categorical(logits=logit).log_prob(sample) for logit, sample in architecture])
    loss = -reward * log_probs.sum()  # REINFORCE update
    loss.backward()
    optimizer.step()

# Reward function (e.g., accuracy - cost)
def get_reward(model, trainloader, testloader, device):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train for a few epochs to evaluate reward
    for epoch in range(5):  # Short training period for evaluation
        model.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Validation performance
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy - 0.01 * sum(p.numel() for p in model.parameters())  # Penalize large models
