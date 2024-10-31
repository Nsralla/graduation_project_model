# This code to train a tcn model only on cls token
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import os
import random
import pandas as pd
# Define TemporalBlock with kernel_size=1
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size=1, dilation=1, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv1d(n_inputs, n_outputs, kernel_size, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, apply_relu=True):
        x = self.conv(x)
        if apply_relu:
            x = self.relu(x)
        x = self.dropout(x)
        return x

# Define TCN without pooling layers
class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, num_classes, kernel_size=1, dropout=0.2):
        super().__init__()
        self.conv1 = TemporalBlock(num_inputs, num_channels[0], kernel_size, dilation=1, dropout=dropout)
        self.conv2 = TemporalBlock(num_channels[0], num_channels[1], kernel_size, dilation=1, dropout=dropout)
        self.conv3 = TemporalBlock(num_channels[1], num_channels[2], kernel_size, dilation=1, dropout=dropout)
        self.fc = nn.Linear(num_channels[2], num_classes)

    def forward(self, x, return_embeddings=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x, apply_relu=False)
        x = torch.mean(x, dim=-1)  # Global mean pooling
        embeddings = x.clone()      # Save embeddings if needed
        logits = self.fc(x)
        if return_embeddings:
            return logits, embeddings
        else:
            return logits

# Custom Dataset class for [CLS] features and labels
class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
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
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Load data, encode labels, and prepare dataset
def load_data(cls_features_path, labels_path, test_size=0.2):
    # Load [CLS] features
    cls_features = torch.load(cls_features_path)  # List of tensors

    # Load labels
    with open(labels_path, 'rb') as f:
        labels = pickle.load(f)

    # Encode labels to integers
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)  # Converts ['A1', 'A2', ...] to integers

    # Stack features and convert labels to tensor
    features = torch.stack(cls_features)  # Shape: [num_samples, hidden_size]
    labels = torch.tensor(labels, dtype=torch.long)  # Ensure labels are of type long for CrossEntropyLoss

    # Split into training and test sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=42, stratify=labels
    )

    # Create Dataset and DataLoader for train and test sets
    train_dataset = MyDataset(X_train, y_train)
    test_dataset = MyDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader, label_encoder

# Function to train TCN model
def train_tcn(model, train_loader, criterion, optimizer, num_epochs=20, device='cuda'):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Add a sequence length dimension
            inputs = inputs.unsqueeze(-1)  # Now shape is [batch_size, 768, 1]

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

# Evaluation function with classification report and confusion matrix
def evaluate_tcn(model, test_loader, label_encoder, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Add a sequence length dimension
            inputs = inputs.unsqueeze(-1)  # Now shape is [batch_size, 768, 1]

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

    # Calculate Additional Metrics
    calculate_additional_metrics(all_labels, all_preds, label_encoder)

# Function to calculate additional metrics: Precision, Recall, F1 Score
def calculate_additional_metrics(all_labels, all_preds, label_encoder):
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

# t-SNE Visualization of Embeddings
def visualize_embeddings(model, test_loader, label_encoder, device='cuda'):
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Add a sequence length dimension
            inputs = inputs.unsqueeze(-1)  # Now shape is [batch_size, 768, 1]

            _, embeddings = model(inputs, return_embeddings=True)
            all_embeddings.append(embeddings.cpu().numpy())
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

# Main function to train, evaluate, and visualize
def main():
    cls_features_path = 'cls_features1.pt'
    labels_path = 'labels1.pkl'

    # Load data
    train_loader, test_loader, label_encoder = load_data(cls_features_path, labels_path)

    # Model configuration
    num_inputs = 768  # [CLS] token size for BERT-base
    num_channels = [256, 128, 64]  # Number of channels for each TemporalBlock
    num_classes = len(label_encoder.classes_)  # Number of unique classes

    # Initialize TCN model
    model = TCN(num_inputs=num_inputs, num_channels=num_channels, num_classes=num_classes, kernel_size=1, dropout=0.2)

    # Define loss criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Train the model
    train_tcn(model, train_loader, criterion, optimizer, num_epochs=20, device=device)

    # Evaluate the model
    evaluate_tcn(model, test_loader, label_encoder, device=device)

    # Visualize Embeddings with t-SNE
    visualize_embeddings(model, test_loader, label_encoder, device=device)
    
    
    # Path where you want to save the model
    model_save_path = "tcn_model.pth"
    # Save only the model's state dictionary
    torch.save(model.state_dict(), model_save_path)
    # Alternative: Save the Entire Model
    # Save the entire model
    torch.save(model, "full_tcn_model.pth")
    



# Run the main function
if __name__ == "__main__":
    main()



# If you want to load the model from the saved state dictionary ()
    # # Initialize the model structure
    # model = TCN(num_inputs=num_inputs, num_channels=num_channels, num_classes=num_classes, kernel_size=1, dropout=0.2)

    # # Load the saved model's state dictionary
    # model.load_state_dict(torch.load(model_save_path))

    # # Set the model to evaluation mode if needed
    # model.eval()
    
#   Or if you want to load the entire model
# Load the entire model
# model = torch.load("full_tcn_model.pth")
# model.eval()


# all_token_features : هاي عبارة عن الفيتسر الكاملة للصوت
# cls_features : هاي عبارة عن الفيتشر الاولى