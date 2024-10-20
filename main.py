import torch
import torch.nn as nn
import os

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        print(f"Before Chomp1d: {x.shape}")
        x = x[:, :, :-self.chomp_size]
        print(f"After Chomp1d: {x.shape}")
        return x

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        print(f"Input to TemporalBlock: {x.shape}")
        out = self.net(x)
        print(f"Output of convs and activations (before residual connection): {out.shape}")
        if self.downsample is not None:
            res = self.downsample(x)
            print(f"Reshaped residual (after downsampling): {res.shape}")
        else:
            res = x
        output = self.relu(out + res)
        print(f"Final output after residual connection and ReLU: {output.shape}")
        return output

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        for i, layer in enumerate(self.network):
            print(f"Passing input through layer {i+1}")
            x = layer(x)
        return x
    


# Ensure the correct file path
url = 'intermediate_results\\processed_features'
file_name = 'feature_0.pt'

# Construct the absolute path
saved_data_path = os.path.join(os.getcwd(), url, file_name)

# Check if the file exists at the constructed path
if os.path.exists(saved_data_path):
    # Load the saved data
    loaded_data = torch.load(saved_data_path)

    # Extract features and labels
    feature = loaded_data['pooled_feature']
    label = loaded_data['label']
    print(feature.shape)
    print(label)
    feature = feature.permute(0,2,1)


# Sample inputs
batch_size = feature.shape[0]
num_channels = feature.shape[1]
sequence_length = feature.shape[2]

# Initialize the TCN model
tcn = TCN(num_inputs=num_channels, num_channels=[16, 32, 64], kernel_size=3, dropout=0.2)

# Create a random input tensor
x = torch.randn(batch_size, num_channels, sequence_length)

# Pass it through the model
output = tcn(x)

print(output.shape)  # Output will have shape (batch_size, 64, sequence_length)
