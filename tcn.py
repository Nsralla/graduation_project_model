import torch
import torch.nn as nn

class TemporalConvNet(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [
                nn.Conv1d(
                    in_channels, 
                    out_channels, 
                    kernel_size, 
                    stride=1, 
                    padding=(kernel_size - 1) * dilation_size, 
                    dilation=dilation_size
                ),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Classifier that includes TCN and a final fully connected layer for scoring
class TCNClassifier(nn.Module):
    def __init__(self, input_size, num_channels, num_classes, kernel_size=3, dropout=0.2):
        super(TCNClassifier, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x, mask=None):
        """
        Forward pass through the TCN and classifier.

        Args:
            x: Tensor of shape [batch_size, input_size, sequence_length]
            mask: Optional Tensor of shape [batch_size, sequence_length]. Elements should be 1 for valid data
                  and 0 for padded data.

        Returns:
            output: Class scores of shape [batch_size, num_classes]
        """
        # Pass through TCN
        tcn_output = self.tcn(x)  # Shape: [batch_size, num_channels[-1], tcn_sequence_length]

        if mask is not None:
            # Resize the mask to match the sequence length of tcn_output
            # Assuming `mask` was originally of shape [batch_size, original_sequence_length]
            batch_size, _, tcn_sequence_length = tcn_output.shape

            # Resize the mask to match the sequence length after passing through the TCN
            mask_resized = torch.nn.functional.interpolate(mask.unsqueeze(1).float(), size=tcn_sequence_length, mode='nearest').squeeze(1)

            # Adjust mask shape to match TCN output
            mask_resized = mask_resized.unsqueeze(1)  # Shape: [batch_size, 1, tcn_sequence_length]

            # Apply mask to the TCN output to zero out padded positions
            tcn_output = tcn_output * mask_resized  # Broadcasting mask over channel dimension

            # Compute the valid length for each sequence (sum over sequence length dimension)
            valid_lengths = mask_resized.sum(dim=2)  # Shape: [batch_size, 1]

            # Avoid division by zero: set any zero lengths to 1 to avoid NaNs
            valid_lengths = valid_lengths.clamp(min=1)

            # Global average pooling over the sequence length while considering the mask
            tcn_output = tcn_output.sum(dim=2) / valid_lengths.squeeze(1)  # Shape: [batch_size, num_channels[-1]]
        else:
            # Global average pooling over the sequence length if no mask is provided
            tcn_output = tcn_output.mean(dim=2)  # Shape: [batch_size, num_channels[-1]]

        # Pass through fully connected layer to get class scores
        output = self.fc(tcn_output)  # Shape: [batch_size, num_classes]

        return output
