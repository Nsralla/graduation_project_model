# Import necessary libraries
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score
from scipy.stats import pearsonr
import torch.nn.functional as F  # Add this import at the top
from sklearn.metrics import cohen_kappa_score  # Add this import at the top
import re



# Set up random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Define the TemporalBlock as provided
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size=3, dilation=1, dropout=0.2):
        """
        Defines a single Temporal Block in the TCN.
        """
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2  # To maintain the sequence length

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, dilation=dilation, padding=padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, dilation=dilation, padding=padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.net = nn.Sequential(
            self.conv1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.relu2,
            self.dropout2
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass for the Temporal Block.
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)  # Residual connection

# Define the TCN as provided with modifications
class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, num_classes, kernel_size=3, dropout=0.2):
        """
        Defines the Temporal Convolutional Network.
        """
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i  # Exponential dilation
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(in_channels, out_channels, kernel_size, dilation=dilation_size, dropout=dropout),
                nn.AvgPool1d(kernel_size=2, stride=2)  # Pooling to reduce sequence length
            ]

        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x, return_features=False):
        """
        Forward pass for the TCN.
        """
        x = self.network(x)
        x = self.global_pool(x).squeeze(-1)  # [batch_size, num_channels]
        logits = self.fc(x)
        if return_features:
            return x, logits  # Return features and logits
        else:
            return logits

# Define the Dataset class for pooled features

import os
import re
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

class PooledFeatureDataset(Dataset):
    def __init__(self, feature_dir, valid_labels):
        """
        Dataset for loading pooled features.

        Args:
            feature_dir (str): Path to the directory containing pooled .npz feature files.
            valid_labels (set): Set of valid label strings.
        """
        self.feature_dir = feature_dir
        self.valid_labels = valid_labels
        self.features = []
        self.labels = []
        self.filenames = []
        self._load_features()

    def _load_features(self):
        """
        Loads features and labels from the .npz files in the feature directory.
        """
        print(f"Loading features from directory: {self.feature_dir}")

        # Prepare a case-insensitive label mapping
        valid_labels_map = {label.lower(): label for label in self.valid_labels}

        for file in tqdm(os.listdir(self.feature_dir), desc=f'Loading {os.path.basename(self.feature_dir)}'):
            if file.endswith('.npz'):
                file_path = os.path.join(self.feature_dir, file)
                try:
                    data = np.load(file_path)
                    feature = data['feature']  # Shape: [pooled_seq_len, hidden_size]
                    filename = str(data['filename'])  # Ensure it's a string
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue

                # Extract label from filename using regex
                parts = re.split(r'[_\s()]', filename)
                label = None
                for part in parts:
                    part_lower = part.lower()
                    if part_lower in valid_labels_map:
                        label = valid_labels_map[part_lower]
                        break

                if label:
                    self.features.append(feature)
                    self.labels.append(label)
                    self.filenames.append(filename)
                else:
                    print(f"Warning: Label not recognized in filename '{filename}'")

        print(f"Total valid samples loaded: {len(self.features)}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """
        Returns:
            feature_tensor (torch.FloatTensor): Pooled feature tensor of shape [hidden_size, pooled_seq_len]
            label (int): Label index
            filename (str): Filename of the sample
        """
        feature = self.features[idx]  # [pooled_seq_len, hidden_size]
        label = self.labels[idx]
        filename = self.filenames[idx]

        # Convert to tensor and transpose to [hidden_size, pooled_seq_len] for TCN input
        feature_tensor = torch.tensor(feature, dtype=torch.float).transpose(0, 1)  # [hidden_size, pooled_seq_len]

        return feature_tensor, label, filename

# Custom collate function to handle varying feature sequence lengths
def collate_fn(batch):
    """
    Collate function to pad features to the maximum sequence length in the batch.

    Args:
        batch (list): List of tuples (feature_tensor, label, filename).

    Returns:
        padded_features (torch.FloatTensor): Padded feature tensors of shape [batch_size, hidden_size, max_seq_len].
        labels (torch.LongTensor): Tensor of label indices.
        filenames (list): List of filenames.
    """
    features, labels, filenames = zip(*batch)
    
    # Determine the maximum sequence length in the batch
    seq_lengths = [f.shape[1] for f in features]
    max_seq_len = max(seq_lengths)
    hidden_size = features[0].shape[0]

    # Initialize a tensor with zeros for padding
    padded_features = torch.zeros(len(features), hidden_size, max_seq_len)

    # Populate the padded_features tensor
    for i, feature in enumerate(features):
        seq_len = feature.shape[1]
        padded_features[i, :, :seq_len] = feature

    # Convert labels to tensor
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return padded_features, labels_tensor, filenames

# Function to encode labels to integers
def encode_labels(labels):
    """
    Encodes string labels to integers.

    Args:
        labels (list): List of string labels.

    Returns:
        label_to_id (dict): Mapping from label string to integer.
        id_to_label (dict): Mapping from integer to label string.
        encoded_labels (list): List of integer-encoded labels.
    """
    unique_labels = sorted(list(set(labels)))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    encoded_labels = [label_to_id[label] for label in labels]
    return label_to_id, id_to_label, encoded_labels

# Function to prepare DataLoaders
def prepare_dataloaders(training_dir, testing_dir, valid_labels, batch_size=16, num_workers=0):
    """
    Prepares DataLoaders for training and testing datasets.

    Args:
        training_dir (str): Path to the training features directory.
        testing_dir (str): Path to the testing features directory.
        valid_labels (set): Set of valid label strings.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
        label_to_id (dict): Label to integer mapping.
        id_to_label (dict): Integer to label mapping.
    """
    # Initialize training dataset
    print("\nInitializing Training Dataset...")
    train_dataset = PooledFeatureDataset(training_dir, valid_labels)
    label_to_id, id_to_label, encoded_train_labels = encode_labels(train_dataset.labels)
    print(f"Training Labels Encoded: {label_to_id}")

    # Update training labels with encoded labels
    train_dataset.labels = encoded_train_labels

    # Initialize testing dataset
    print("\nInitializing Testing Dataset...")
    test_dataset = PooledFeatureDataset(testing_dir, valid_labels)
    # Ensure testing labels use the same encoding as training
    test_encoded_labels = []
    for label in test_dataset.labels:
        if label in label_to_id:
            test_encoded_labels.append(label_to_id[label])
        else:
            print(f"Warning: Testing label '{label}' not found in training labels.")
            test_encoded_labels.append(-1)  # Assign -1 for unknown labels
    test_dataset.labels = test_encoded_labels

    # Create DataLoaders with num_workers=0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True,
        collate_fn=collate_fn
    )

    return train_loader, test_loader, label_to_id, id_to_label

# Training function
def train_epoch(model, device, train_loader, criterion, optimizer, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The TCN model.
        device (torch.device): Device to run the model on.
        train_loader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        epoch (int): Current epoch number.

    Returns:
        avg_loss (float): Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    for batch_idx, (features, labels, _) in enumerate(tqdm(train_loader, desc=f'Training Epoch {epoch}')):
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch}] Average Loss: {avg_loss:.4f}")
    return avg_loss

# Evaluation function with t-SNE plotting
def evaluate(model, device, test_loader, id_to_label, epoch, save_dir='tsne_plots'):
    """
    Evaluates the model on the test dataset and plots t-SNE.

    Args:
        model (nn.Module): The trained TCN model.
        device (torch.device): Device to run the model on.
        test_loader (DataLoader): DataLoader for testing data.
        id_to_label (dict): Mapping from integer labels to label strings.
        epoch (int): Current epoch number.
        save_dir (str): Directory to save t-SNE plots.
    
    Returns:
        accuracy (float): Overall accuracy on the test set.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_features = []
    all_outputs = []  # Initialize list to collect model outputs


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for features, labels, _ in tqdm(test_loader, desc='Evaluating'):
            features, labels = features.to(device), labels.to(device)
            features_output, outputs = model(features, return_features=True)
            _, preds = torch.max(outputs, 1)

            # Filter out unknown labels (-1)
            mask = labels != -1
            preds = preds[mask]
            labels = labels[mask]
            features_output = features_output[mask]
            outputs = outputs[mask]  # Apply mask to outputs
             # Apply softmax to outputs to get probabilities
            probabilities = F.softmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_features.append(features_output.cpu().numpy())
            all_outputs.append(probabilities.cpu().numpy())  # Collect probabilities


    # Concatenate all features
    all_features = np.concatenate(all_features, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)  # Shape: [n_samples, n_classes]


    # Apply t-SNE
    print("Applying t-SNE on the collected features...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(all_features)

    # Create a DataFrame for plotting
    df_tsne = pd.DataFrame({
        'tsne-2d-one': tsne_results[:,0],
        'tsne-2d-two': tsne_results[:,1],
        'label': [id_to_label[label] for label in all_labels]
    })

    # Plot t-SNE
    plt.figure(figsize=(10,8))
    sns.scatterplot(
        x='tsne-2d-one', y='tsne-2d-two',
        hue='label',
        palette=sns.color_palette("hsv", len(id_to_label)),
        data=df_tsne,
        legend="full",
        alpha=0.7
    )
    plt.title(f't-SNE Visualization after Epoch {epoch}')
    plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f'tsne_epoch_{epoch}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"t-SNE plot saved to {plot_path}")

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=[id_to_label[i] for i in range(len(id_to_label))]))
    
     # Mean Absolute Error (MAE)
    mae = mean_absolute_error(all_labels, all_preds)
    print(f"Mean Absolute Error: {mae:.4f}")

    # Correlation (Pearson)
    if len(set(all_labels)) > 1:  # Ensure there is more than one unique label
        correlation, _ = pearsonr(all_labels, all_preds)
        print(f"Pearson Correlation: {correlation:.4f}")

    # AUC (Area Under the Curve) - One-vs-all AUC for multi-class
     # AUC (Area Under the Curve) - One-vs-all AUC for multi-class
    try:
        auc = roc_auc_score(all_labels, all_outputs, multi_class='ovr')
        print(f"AUC (One-vs-all): {auc:.4f}")
    except ValueError as e:
        print(f"AUC could not be calculated: {e}")
        
     # Weighted Kappa Score (Cohen's Kappa with Quadratic Weights)
    try:
        weighted_kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
        print(f"Weighted Kappa Score (Quadratic): {weighted_kappa:.4f}")
    except ValueError as e:
        print(f"Weighted Kappa could not be calculated: {e}")
    
        
    
    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=[id_to_label[i] for i in range(len(id_to_label))]))


    # Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=[id_to_label[i] for i in range(len(id_to_label))],
                yticklabels=[id_to_label[i] for i in range(len(id_to_label))], 
                cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix after Epoch {epoch}')
    plt.tight_layout()
    cm_plot_path = os.path.join(save_dir, f'confusion_matrix_epoch_{epoch}.png')
    plt.savefig(cm_plot_path)
    plt.close()
    print(f"Confusion matrix plot saved to {cm_plot_path}")

    return accuracy

# Main function
def main():
    # Define paths to the averaged feature directories
    training_pooled_dir = r'Youtube\extracted features\training features'  # Update if necessary
    testing_pooled_dir = r'Youtube\extracted features\testing features'    # Update if necessary

    # Define valid labels
    valid_labels = {'A1', 'C1', 'C2'}

    # Prepare DataLoaders with num_workers=0
    print("\n--- Preparing DataLoaders ---")
    train_loader, test_loader, label_to_id, id_to_label = prepare_dataloaders(
        training_pooled_dir,
        testing_pooled_dir,
        valid_labels,
        batch_size=3,  # Adjust based on GPU memory
        num_workers=0    # Set to 0 to avoid multiprocessing issues
    )

    # Check if there are any training samples
    if len(train_loader.dataset) == 0:
        print("No training samples found. Exiting.")
        return

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Initialize the TCN model
    # Determine hidden_size from the first training sample
    sample_feature, _, _ = train_loader.dataset[0]
    hidden_size = sample_feature.shape[0]
    print(f"Hidden Size: {hidden_size}")

    num_channels = [64, 128, 256]  # Example channel sizes, adjust as needed
    num_classes = len(label_to_id)
    kernel_size = 3
    dropout = 0.2

    model = TCN(
        num_inputs=hidden_size,  # hidden_size
        num_channels=num_channels,
        num_classes=num_classes,
        kernel_size=kernel_size,
        dropout=dropout
    ).to(device)

    print("\nTCN Model Initialized:")
    print(model)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 6  # Adjust based on convergence

    best_accuracy = 0.0
    for epoch in range(1, num_epochs + 1):
        print(f"\n=== Epoch {epoch} ===")
        train_loss = train_epoch(model, device, train_loader, criterion, optimizer, epoch)
        
        # Optionally, add validation here if you have a validation set

        # Evaluate on test set and plot t-SNE
        test_accuracy = evaluate(model, device, test_loader, id_to_label, epoch)

        # Save the model if it has the best accuracy so far
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            model_save_path = './best_tcn_model.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved with accuracy: {best_accuracy * 100:.2f}%")

    print("\nTraining complete.")
    print(f"Best Test Accuracy: {best_accuracy * 100:.2f}%")

# Entry point
if __name__ == '__main__':
    main()
