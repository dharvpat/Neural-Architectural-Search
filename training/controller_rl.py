import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class RLController(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(RLController, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x.unsqueeze(0), hidden)
        x = self.fc(x.squeeze(0))
        return x, hidden

    def sample(self, input_dim):
        start = torch.zeros(input_dim)  # Initial input
        logits, hidden = self.forward(start)
        return Categorical(logits=logits).sample(), hidden