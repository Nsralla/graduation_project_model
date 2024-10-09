import torch
from torch.utils.data import Dataset

# Define class labels and create a mapping from labels to indices
class_labels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
label_to_index = {label: idx for idx, label in enumerate(class_labels)}

class IELTSDataSetForCLS(Dataset):
    def __init__(self, cls_tokens):
        # Store raw data (features and labels) without converting to tensors
        self.features = [item['concatenated_cls'] for item in cls_tokens]
        self.labels = [label_to_index[item['label']] for item in cls_tokens]  # Map string labels to indices
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Convert the features and labels to tensors on-the-fly
        feature = torch.tensor(self.features[idx], dtype=torch.float32)  # Ensure features are float tensors
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # Ensure labels are integer tensors (long for classification)
        return feature, label
