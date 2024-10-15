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

# Step 2: Iterate through the DataLoader and process each sample
window_size = 5
stride_size = 8

for idx, (feature, label) in enumerate(dataloader):
    # Apply pooling to the feature
    feature = feature.squeeze(0)  # Remove the extra batch dimension
    pooled_feature = F.avg_pool1d(feature.transpose(1, 2), kernel_size=window_size, stride=stride_size)
    pooled_feature = pooled_feature.transpose(1, 2)
    
    # Create a unique file name for each feature
    output_file_name = os.path.join(output_directory, f'feature_{idx}.pt')
    
    # Save the pooled feature and corresponding label in a dictionary
    torch.save({'pooled_feature': pooled_feature, 'label': label.squeeze(0)}, output_file_name)

    print(f'Saved: {output_file_name}')

print("All features processed and saved!")
