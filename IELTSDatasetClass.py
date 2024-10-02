import torch
from torch.utils.data import Dataset

class IELTSDataset(Dataset):
    def __init__(self, features, labels):
        """
        Args:
            features (list of Tensors): A list of tensors representing the features, each of shape [sequence_length, feature_dim].
            labels (list of int): A list of labels corresponding to each feature tensor.
        """
        self.features = features
        self.labels = labels

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.features)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to be fetched.

        Returns:
            dict: A dictionary containing 'feature' and 'label' for the given index.
        """
        # Fetch the feature and label for the given index
        feature = self.features[idx]
        label = self.labels[idx]
        
        # Convert feature and label to torch tensors if needed
        if not isinstance(feature, torch.Tensor):
            feature = torch.tensor(feature, dtype=torch.float32)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)

        return {'feature': feature, 'label': label}



    
import re

def extract_label_from_path(file_path):
    """Extracts the score from the file path."""
    # Use a regex to match the score pattern (A1, A2, B1, B2, C1, C2)
    match = re.search(r'([A-Ca-c][1-2])', file_path)
    if match:
        score = match.group(1).upper()  # Extract and convert to uppercase
        print("Collected score:", score)
        return score
    else:
        raise ValueError(f"Score not found in the file path: {file_path}")

# Examples:
# audios\manually\SM_THA_PTJ2_047_B2_0.mp3 -> B2
# audios\manually\video133_c2.mp3 -> C2
