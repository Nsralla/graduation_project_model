import torch
import torch.nn as nn

class FullyConnectedClassifier(nn.Module):
    def __init__(self, input_size, num_hidden_units, num_classes):
        super(FullyConnectedClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, num_hidden_units)  # Fully connected layer 1
        self.relu = nn.ReLU()  # ReLU activation
        self.fc2 = nn.Linear(num_hidden_units, num_classes)  # Fully connected layer 2

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
