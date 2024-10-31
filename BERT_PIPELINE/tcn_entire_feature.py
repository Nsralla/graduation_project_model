# import torch
# # Load the saved all token features
# all_token_features = torch.load('all_token_features1.pt')


# # print the shape of the first embeddings
# print(all_token_features[123].shape)
# print(all_token_features[1].shape)
# print(all_token_features[2].shape)
# print(all_token_features[3].shape)
# print(all_token_features[4].shape)

# This code to train a tcn model on the entire feature not only on cls token

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import os
import random
import pandas as pd

# ----------------------------
# 1. Setting Random Seeds for Reproducibility
# ----------------------------

def set_seed(seed=42):
    """
    Set the random seed for reproducibility.
    
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU.
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------
# 2. Defining the TemporalBlock and TCN Classes
# ----------------------------

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size=3, dilation=1, dropout=0.2):
        """
        Defines a single Temporal Block in the TCN.
        
        Args:
            n_inputs (int): Number of input channels.
            n_outputs (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            dilation (int): Dilation rate for the convolution.
            dropout (float): Dropout rate.
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
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, n_inputs, seq_length].
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, n_outputs, seq_length].
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)  # Residual connection

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, num_classes, kernel_size=3, dropout=0.2):
        """
        Defines the Temporal Convolutional Network.
        
        Args:
            num_inputs (int): Number of input channels (e.g., feature size).
            num_channels (list of int): Number of channels in each Temporal Block.
            num_classes (int): Number of output classes.
            kernel_size (int): Size of the convolutional kernel.
            dropout (float): Dropout rate.
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
    
    def forward(self, x):
        """
        Forward pass for the TCN.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_inputs, seq_length].
        
        Returns:
            torch.Tensor: Logits tensor of shape [batch_size, num_classes].
        """
        x = self.network(x)
        x = self.global_pool(x).squeeze(-1)  # [batch_size, num_channels]
        logits = self.fc(x)
        return logits

# ----------------------------
# 3. Custom Dataset Class
# ----------------------------

class MyDataset(Dataset):
    def __init__(self, features, labels):
        """
        Custom Dataset for loading token features and labels.
        
        Args:
            features (list of torch.Tensor): List of tensors, each of shape [seq_len, feature_size].
            labels (torch.Tensor): Tensor of labels.
        """
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ----------------------------
# 4. Custom Collate Function for DataLoader
# ----------------------------

def collate_fn(batch):
    """
    Custom collate function to pad sequences in a batch.
    
    Args:
        batch (list of tuples): Each tuple contains (feature_tensor, label).
    
    Returns:
        tuple: (padded_features, labels)
            - padded_features: Tensor of shape [batch_size, feature_size, max_seq_len]
            - labels: Tensor of shape [batch_size]
    """
    features, labels = zip(*batch)  # Unzip the batch
    # Pad sequences
    padded_features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)  # [batch_size, max_seq_len, feature_size]
    # Permute to [batch_size, feature_size, max_seq_len] for Conv1D
    padded_features = padded_features.permute(0, 2, 1)
    labels = torch.stack(labels)
    return padded_features, labels

# ----------------------------
# 5. Loading Data
# ----------------------------

def load_data(features_path, labels_path, test_size=0.2, seed=42):
    """
    Load features and labels, encode labels, and prepare DataLoaders.
    
    Args:
        features_path (str): Path to the token features file.
        labels_path (str): Path to the labels file.
        test_size (float): Proportion of the dataset to include in the test split.
        seed (int): Random seed for reproducibility.
    
    Returns:
        tuple: (train_loader, test_loader, label_encoder)
    """
    # Load all token features
    all_token_features = torch.load(features_path)  # List of tensors, each of shape [1, seq_len, 768]
    
    # Process features: remove the batch dimension
    processed_features = [f.squeeze(0) for f in all_token_features]  # List of [seq_len, 768]
    
    # Load labels
    with open(labels_path, 'rb') as f:
        labels = pickle.load(f)
    
    # Encode labels to integers
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)  # Converts ['A1', 'A2', ...] to integers
    
    # Convert to tensors
    labels_tensor = torch.tensor(encoded_labels, dtype=torch.long)
    
    # Split into training and test sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        processed_features,
        labels_tensor,
        test_size=test_size,
        random_state=seed,
        stratify=labels_tensor
    )
    
    # Create Dataset and DataLoader
    train_dataset = MyDataset(X_train, y_train)
    test_dataset = MyDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, test_loader, label_encoder

# ----------------------------
# 6. Training Function
# ----------------------------

def train_tcn(model, train_loader, criterion, optimizer, num_epochs=20, device='cuda'):
    """
    Train the TCN model.
    
    Args:
        model (nn.Module): The TCN model.
        train_loader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        num_epochs (int): Number of training epochs.
        device (str): Device to train on ('cuda' or 'cpu').
    """
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# ----------------------------
# 7. Evaluation Function
# ----------------------------

def evaluate_tcn(model, test_loader, label_encoder, device='cuda'):
    """
    Evaluate the TCN model on the test set.
    
    Args:
        model (nn.Module): The trained TCN model.
        test_loader (DataLoader): DataLoader for test data.
        label_encoder (LabelEncoder): Label encoder to decode predictions.
        device (str): Device to evaluate on ('cuda' or 'cpu').
    """
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification Report
    class_names = label_encoder.classes_
    print("Classification Report:\n", classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    cm_display.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.show()

    # Additional Metrics
    calculate_additional_metrics(all_labels, all_preds, label_encoder)

def calculate_additional_metrics(all_labels, all_preds, label_encoder):
    """
    Calculate and display additional classification metrics.
    
    Args:
        all_labels (list): True labels.
        all_preds (list): Predicted labels.
        label_encoder (LabelEncoder): Label encoder to decode class indices.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=np.arange(len(label_encoder.classes_)), zero_division=0
    )

    # Create a DataFrame for better visualization
    metrics_df = pd.DataFrame({
        'Class': label_encoder.classes_,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })

    print("\nAdditional Classification Metrics:")
    print(metrics_df)

# ----------------------------
# 8. Visualization Function
# ----------------------------

def visualize_embeddings(model, test_loader, label_encoder, device='cuda'):
    """
    Visualize the embeddings of the test set using t-SNE.
    
    Args:
        model (nn.Module): The trained TCN model.
        test_loader (DataLoader): DataLoader for test data.
        label_encoder (LabelEncoder): Label encoder to decode labels.
        device (str): Device to perform computations on ('cuda' or 'cpu').
    """
    model.to(device)
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass to get embeddings (after global pooling)
            # Modify the TCN to return embeddings if needed, or extract before the final FC layer
            x = model.network(inputs)
            x = model.global_pool(x).squeeze(-1)  # [batch_size, num_channels]
            all_embeddings.append(x.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Stack all embeddings
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.array(all_labels)

    # t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # Plot t-SNE
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embeddings_2d[:, 0], y=embeddings_2d[:, 1],
        hue=all_labels, palette='tab10', legend='full', alpha=0.7
    )
    plt.title("t-SNE Visualization of TCN Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Classes", labels=label_encoder.classes_, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# ----------------------------
# 9. Saving and Loading the Model
# ----------------------------

def save_model(model, optimizer, label_encoder, filename='all_feature_tcn_model.pth'):
    """
    Save the model's state dictionary along with optimizer and label encoder.
    
    Args:
        model (nn.Module): Trained model.
        optimizer (torch.optim.Optimizer): Optimizer used during training.
        label_encoder (LabelEncoder): Label encoder used to encode labels.
        filename (str): File path to save the model.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'label_encoder': label_encoder
    }
    torch.save(checkpoint, filename)
    print(f"Model saved to {filename}")


# ----------------------------
# 10. Main Function
# ----------------------------

def main():
    # ----------------------------
    # a. Set the Seed
    # ----------------------------
    set_seed(42)
    
    # ----------------------------
    # b. Paths to Data Files
    # ----------------------------
    features_path = 'all_token_features1.pt'  # Path to the token features
    labels_path = 'labels1.pkl'              # Path to the labels
    
    # ----------------------------
    # c. Load Data
    # ----------------------------
    train_loader, test_loader, label_encoder = load_data(
        features_path=features_path,
        labels_path=labels_path,
        test_size=0.2,
        seed=42
    )
    
    # ----------------------------
    # d. Model Configuration
    # ----------------------------
    num_inputs = 768  # Feature size from BERT
    num_channels = [256, 128, 64]  # Channels in each Temporal Block
    num_classes = len(label_encoder.classes_)  # Number of classes
    
    # ----------------------------
    # e. Initialize Model, Criterion, and Optimizer
    # ----------------------------
    model = TCN(
        num_inputs=num_inputs,
        num_channels=num_channels,
        num_classes=num_classes,
        kernel_size=3,  # Standard kernel size for TCN
        dropout=0.2
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # ----------------------------
    # f. Determine Device
    # ----------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ----------------------------
    # g. Train the Model
    # ----------------------------
    train_tcn(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=10,
        device=device
    )
    
    # ----------------------------
    # h. Evaluate the Model
    # ----------------------------
    evaluate_tcn(
        model=model,
        test_loader=test_loader,
        label_encoder=label_encoder,
        device=device
    )
    
    # ----------------------------
    # i. Visualize Embeddings
    # ----------------------------
    visualize_embeddings(
        model=model,
        test_loader=test_loader,
        label_encoder=label_encoder,
        device=device
    )
    
    # ----------------------------
    # j. Save the Trained Model
    # ----------------------------
    save_model(
        model=model,
        optimizer=optimizer,
        label_encoder=label_encoder,
        filename='full_feaature_tcn_model.pth'
    )

# ----------------------------
# 11. Running the Main Function
# ----------------------------

if __name__ == "__main__":
    main()

