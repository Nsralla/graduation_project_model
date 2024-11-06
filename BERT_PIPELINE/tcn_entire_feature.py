import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
    cohen_kappa_score,
    mean_absolute_error,
)
from scipy.stats import pearsonr

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ----------------------------
# 1. Setting Random Seeds for Reproducibility
# ----------------------------

def set_seed(seed=42):
    """
    Set the random seed for reproducibility.
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

    def forward(self, x):
        """
        Forward pass for the TCN.
        """
        x = self.network(x)
        x = self.global_pool(x).squeeze(-1)  # [batch_size, num_channels]
        logits = self.fc(x)
        return logits

# ----------------------------
# 3. Label Extraction Function
# ----------------------------

def extract_label_from_filename(filename):
    filename = filename.lower()
    possible_labels = ['a2', 'b1_1', 'b1_2', 'b2']

    # Check if any known label is in the filename
    for label in possible_labels:
        if label in filename:
            return label

    # If no label is found, return None
    print(f"No label found in filename: {filename}")
    return None

# ----------------------------
# 4. Custom Dataset Class
# ----------------------------

class MyDataset(Dataset):
    def __init__(self, features, labels):
        """
        Custom Dataset for loading token features and labels.
        """
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ----------------------------
# 5. Custom Collate Function for DataLoader
# ----------------------------

def collate_fn(batch):
    """
    Custom collate function to pad sequences in a batch.
    """
    features, labels = zip(*batch)  # Unzip the batch
    # Pad sequences
    padded_features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)  # [batch_size, max_seq_len, feature_size]
    # Permute to [batch_size, feature_size, max_seq_len] for Conv1D
    padded_features = padded_features.permute(0, 2, 1)
    labels = torch.stack(labels)
    return padded_features, labels

# ----------------------------
# 6. Loading Data
# ----------------------------

def load_features_and_labels_from_folder(folder_path):
    """
    Load features and labels from a given folder.
    """
    features = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pt'):
            filepath = os.path.join(folder_path, filename)
            # Load the feature tensor
            feature_tensor = torch.load(filepath)  # Assuming it's a tensor
            features.append(feature_tensor)
            # Extract the label from filename
            label = extract_label_from_filename(filename)
            if label is not None:
                labels.append(label)
            else:
                # Handle the case where no label is found
                print(f"Label not found for file: {filename}")
                # Remove the corresponding feature since we don't have a label
                features.pop()
    return features, labels

def load_data(training_features_dir, testing_features_dir):
    """
    Load features and labels, encode labels, and prepare DataLoaders.
    """
    # Load features and labels from training folder
    train_features, train_labels = load_features_and_labels_from_folder(training_features_dir)

    # Load features and labels from testing folder
    test_features, test_labels = load_features_and_labels_from_folder(testing_features_dir)

    # Combine labels to fit the LabelEncoder
    all_labels = train_labels + test_labels

    # Encode labels to integers
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    encoded_train_labels = label_encoder.transform(train_labels)
    encoded_test_labels = label_encoder.transform(test_labels)

    # Convert labels to tensors
    train_labels_tensor = torch.tensor(encoded_train_labels, dtype=torch.long)
    test_labels_tensor = torch.tensor(encoded_test_labels, dtype=torch.long)

    # Create Dataset and DataLoader
    train_dataset = MyDataset(train_features, train_labels_tensor)
    test_dataset = MyDataset(test_features, test_labels_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_loader, test_loader, label_encoder

# ----------------------------
# 7. Training Function
# ----------------------------

def train_tcn(model, train_loader, criterion, optimizer, num_epochs=20, device='cuda'):
    """
    Train the TCN model.
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
# 8. Evaluation Function
# ----------------------------

def evaluate_tcn(model, test_loader, label_encoder, device='cuda'):
    """
    Evaluate the TCN model on the test set.
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
    Calculate and display additional classification metrics including WK, Corr, and MAE.
    """
    # Calculate Precision, Recall, and F1-Score
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=np.arange(len(label_encoder.classes_)), zero_division=0
    )

    # Weighted Kappa (WK)
    wk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")

    # Correlation (Corr)
    corr, _ = pearsonr(all_labels, all_preds)

    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(all_labels, all_preds)

    # Display the metrics
    print(f"\nWeighted Kappa (WK): {wk:.4f}")
    print(f"Correlation (Corr): {corr:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

    # Create a DataFrame for better visualization of class-specific metrics
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
# 9. Visualization Function
# ----------------------------

def visualize_embeddings(model, data_loader, label_encoder, device='cuda'):
    """
    Visualize the embeddings of the dataset using t-SNE.
    """
    model.to(device)
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass to get embeddings (after global pooling)
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
# 10. Saving the Model
# ----------------------------

def save_model(model, optimizer, label_encoder, filename='full_features_tcn_model_icnale_dataset.pth'):
    """
    Save the model's state dictionary along with optimizer and label encoder.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'label_encoder': label_encoder
    }
    torch.save(checkpoint, filename)
    print(f"Model saved to {filename}")

# ----------------------------
# 11. Main Function
# ----------------------------

def main():
    # Set the Seed
    set_seed(42)

    # Paths to data directories
    training_features_dir = 'training_features_Icnale_base_model'
    testing_features_dir = 'testing_features_Icnale_base_model'

    # Load Data
    train_loader, test_loader, label_encoder = load_data(
        training_features_dir=training_features_dir,
        testing_features_dir=testing_features_dir
    )

    # Model Configuration
    num_inputs = 768  # Feature size from BERT
    num_channels = [256, 128, 64]  # Channels in each Temporal Block
    num_classes = len(label_encoder.classes_)  # Number of classes

    # Initialize Model, Criterion, and Optimizer
    model = TCN(
        num_inputs=num_inputs,
        num_channels=num_channels,
        num_classes=num_classes,
        kernel_size=3,  # Standard kernel size for TCN
        dropout=0.2
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Determine Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Train the Model
    train_tcn(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=8,
        device=device
    )

    # Evaluate the Model
    evaluate_tcn(
        model=model,
        test_loader=test_loader,
        label_encoder=label_encoder,
        device=device
    )

    # Visualize Embeddings on Test Set
    visualize_embeddings(
        model=model,
        data_loader=test_loader,
        label_encoder=label_encoder,
        device=device
    )

    # Save the Trained Model
    save_model(
        model=model,
        optimizer=optimizer,
        label_encoder=label_encoder,
        filename='full_features_tcn_model_icnale_dataset.pth'
    )

# ----------------------------
# 12. Running the Main Function
# ----------------------------

if __name__ == "__main__":
    main()
