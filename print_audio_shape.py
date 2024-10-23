import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os

# Create a folder for the output if it doesn't exist
output_directory = 'intermediate_results\\processed_features\\'
os.makedirs(output_directory, exist_ok=True)

class LargeFeatureDataset(Dataset):
    def __init__(self, data_path):
        # Load the data (assuming the file contains a dictionary with 'features' and 'labels')
        self.data = torch.load(data_path)
        self.features = self.data['features']
        self.labels = self.data['labels']
        self.total_samples = len(self.features)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Return the feature and corresponding label
        return self.features[idx], self.labels[idx]

# Step 1: Initialize the dataset and dataloader
data_path = 'intermediate_results\\features_labels.pt'
dataset = LargeFeatureDataset(data_path)

# Create a DataLoader to load in smaller batches
batch_size = 1  # Load one sample at a time
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Define pooling parameters
window_size = 5
stride_size = 8

# Step 2: Print the shape of the processed features
for idx, (feature, label) in enumerate(dataloader):
    # Adjust the shape for avg_pool1d
    feature = feature.squeeze(0)  # Remove the extra batch dimension to get shape [1, 2107, 1024]
    feature = feature.transpose(1, 2)  # Reshape to [1, 1024, 2107] for avg_pool1d

    # Apply pooling to the feature
    pooled_feature = F.avg_pool1d(feature, kernel_size=window_size, stride=stride_size)

    # Reshape back to [1, Pooled Length, 1024]
    pooled_feature = pooled_feature.transpose(1, 2)

    # Print the shape of the pooled feature
    print(f"Sample {idx + 1} - Pooled Features shape: {pooled_feature.shape}")

    # Stop after processing 50 samples
    if idx >= 49:
        break
