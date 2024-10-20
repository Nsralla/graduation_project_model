import torch
import torch.nn as nn
import os
from padding import logger

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2  # Correct padding to maintain sequence length
        self.conv_block = nn.Sequential(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        logger.debug(f"Input shape to TemporalBlock: {x.shape}")
        out = self.conv_block(x)
        logger.debug(f"Output shape after conv_block in TemporalBlock: {out.shape}")
        res = x if self.downsample is None else self.downsample(x)
        logger.debug(f"Residual connection shape in TemporalBlock: {res.shape}")
        logger.debug(f"______________________________________________________");
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, num_classes, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            layers.append(TemporalBlock(
                num_inputs if i == 0 else num_channels[i-1],
                num_channels[i],
                kernel_size,
                dilation=2**i,
                dropout=dropout
            ))
        self.network = nn.Sequential(*layers)
        
        self.pooling_layer = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        logger.debug(f"Input shape to TCN: {x.shape}")
        x = self.network(x)  # Pass through TCN layers
        logger.debug(f"Shape after TCN layers: {x.shape}")
        
        seq_len = x.shape[2]
        if seq_len > 40:
            pooling_kernel_size = min(5, seq_len // 2)
            self.pooling_layer = nn.MaxPool1d(kernel_size=pooling_kernel_size)
            x = self.pooling_layer(x)
            logger.debug(f"Shape after pooling layer: {x.shape}")
        
        x = torch.mean(x, dim=-1)
        logger.debug(f"Shape after global pooling: {x.shape}")
        
        x = self.fc(x)
        logger.debug(f"Shape after fully connected layer: {x.shape}")
        return torch.softmax(x, dim=-1)

# Ensure the correct file path
url = 'intermediate_results\\processed_features'
file_name = 'feature_0.pt'
saved_data_path = os.path.join(os.getcwd(), url, file_name)

if os.path.exists(saved_data_path):
    loaded_data = torch.load(saved_data_path)
    feature = loaded_data['pooled_feature']
    label = loaded_data['label']
    logger.debug(f"Loaded feature shape: {feature.shape}")
    logger.debug(f"Loaded label: {label}")
    feature = feature.permute(0, 2, 1)

num_classes = 7  # A1, A2, B1_1, B1_2, B2, C1, C2
batch_size = feature.shape[0]
sequence_length = feature.shape[2]

# Initialize the TCN model
tcn = TCN(num_inputs=feature.shape[1], num_channels=[32, 64, 128], num_classes=num_classes, kernel_size=3, dropout=0.2)

# Create a random input tensor
x = torch.randn(batch_size, feature.shape[1], sequence_length)
logger.debug(f"Random input tensor shape: {x.shape}")

# Pass it through the model
output = tcn(x)
logger.debug(f"Output shape: {output.shape}")
print(output.shape)
print(output)








# Suppose:

# Input x has 64 channels.
# The convolutional layers in TemporalBlock output 128 channels.
# Since the number of channels does not match, self.downsample will be defined as a Conv1d layer with in_channels=64 and out_channels=128:

# python
# Copy code
# self.downsample = nn.Conv1d(64, 128, 1)  # Adjusts input channels to match output channels
# The line:

# python
# Copy code
# res = x if self.downsample is None else self.downsample(x)
# will execute self.downsample(x), transforming x to have 128 channels, so it can be added to the output of the convolutional layers.