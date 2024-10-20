import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import random


# Custom collate function
def my_collate(batch):
    features, labels = zip(*batch)
    max_channels = max(f.shape[1] for f in features)
    padded_features = [F.pad(f, (0, 0, 0, max_channels - f.shape[1])) for f in features]
    padded_features = torch.stack(padded_features)
    padded_features = padded_features.squeeze(1).permute(0, 2, 1)
    labels = torch.tensor(labels)
    return padded_features, labels

# Custom Dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Define the TCN architecture
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TCNBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) * dilation, dilation=dilation)
        self.batch_norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        return F.relu(x)

class TCNClassifier(nn.Module):
    def __init__(self, input_size, num_channels, num_classes, kernel_size=4, dropout=0.2):
        super(TCNClassifier, self).__init__()
        layers = []
        num_levels = len(num_channels)

        # Input layer
        layers.append(TCNBlock(input_size, num_channels[0], kernel_size, dilation=1))

        # Intermediate TCN layers
        for i in range(1, num_levels):
            dilation_size = 2 ** i
            layers.append(TCNBlock(num_channels[i - 1], num_channels[i], kernel_size, dilation=dilation_size))

        self.network = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        x = self.network(x)
        x = torch.mean(x, dim=2)  # Global average pooling across the time dimension
        x = self.dropout(x)
        return self.fc(x)

# Focal Loss for class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# Training function with gradient accumulation and learning rate scheduler
def train_model_with_early_stopping(model, train_loader, test_loader, num_epochs=30, learning_rate=0.001, patience=5, accumulation_steps=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    best_accuracy = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels) / accumulation_steps  # Normalize loss for accumulation

            loss.backward()
            running_loss += loss.item()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

        # Evaluate the model on the test set
        current_accuracy = test_model(model, test_loader)

        # # Early stopping check
        # if current_accuracy > best_accuracy:
        #     best_accuracy = current_accuracy
        #     patience_counter = 0  # Reset the counter
        # else:
        #     patience_counter += 1
        #     if patience_counter >= patience:
        #         print(f"Early stopping at epoch {epoch+1}, best accuracy: {best_accuracy:.2f}%")
        #         break

        scheduler.step(running_loss / len(train_loader))

# Testing function with confusion matrix
# Testing function with confusion matrix
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    y_pred = []
    y_true = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Generate and display confusion matrix
    # conf_matrix = confusion_matrix(y_true, y_pred)
    # plt.figure(figsize=(10, 7))
    # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    # plt.ylabel('Actual')
    # plt.xlabel('Predicted')
    # plt.show()

    return accuracy  # Return accuracy so that it can be compared in training


# Weight initialization
def init_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

# Limit the number of B1_2 samples to 500
def reduce_class_B1_2(features_list, labels, target_class='B1_2', max_samples=500):
    # Find indices for the B1_2 class
    target_class_idx = class_labels.index(target_class)
    target_indices = [i for i, label in enumerate(labels) if label == target_class_idx]

    # Randomly sample 500 indices from B1_2
    if len(target_indices) > max_samples:
        selected_indices = random.sample(target_indices, max_samples)
    else:
        selected_indices = target_indices  # If less than 500, take all

    # Get all other class indices
    other_indices = [i for i, label in enumerate(labels) if label != target_class_idx]

    # Combine selected B1_2 indices with other class indices
    final_indices = selected_indices + other_indices

    # Filter the features and labels based on final indices
    reduced_features = [features_list[i] for i in final_indices]
    reduced_labels = [labels[i] for i in final_indices]

    return reduced_features, reduced_labels
# Main code
# Filter the dataset to only include samples with labels C1, C2, A1
def filter_dataset_by_labels(features_list, labels, target_labels):
    target_indices = [i for i, label in enumerate(labels) if label in target_labels]

    # Filter features and labels based on the selected indices
    filtered_features = [features_list[i] for i in target_indices]
    filtered_labels = [labels[i] for i in target_indices]

    return filtered_features, filtered_labels
# Ensure the correct file path
url = 'process_text_audio_seperatly\\text_features_only'
file_name = 'text_features_and_labels.pt'

# Construct the absolute path
saved_data_path = os.path.join(os.getcwd(), url, file_name)
class_labels = ['A1','A2','B2','C1', 'C2']
input_size = 1024
num_channels = [32, 64, 128]  # Increased channels for better capacity
num_classes = len(class_labels)
# Main code
# Main code
if os.path.exists(saved_data_path):
    loaded_data = torch.load(saved_data_path)
    features_list = loaded_data['features']
    labels = loaded_data['labels']

    # Define the target labels to keep: C1, C2, A1
    target_labels = [class_labels.index('C1'), class_labels.index('C2'), class_labels.index('A1'), class_labels.index('A2'),class_labels.index('B2')]

    # Filter the dataset to keep only samples with the specified labels
    filtered_features, filtered_labels = filter_dataset_by_labels(features_list, labels, target_labels)

    # Create a new dataset with the filtered data
    dataset = MyDataset(filtered_features, filtered_labels)

    # Split the dataset into train and test sets
    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Extract labels from the train_dataset for sampling
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]

    # Compute class counts based on the training labels
    class_counts = torch.bincount(torch.tensor(train_labels))
    class_weights = 1. / class_counts.float()

    # Assign sample weights based on train_dataset labels
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # Create data loaders with WeightedRandomSampler for train_loader
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, collate_fn=my_collate)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=my_collate)

    # Initialize and train the model
    model = TCNClassifier(input_size=input_size, num_channels=num_channels, num_classes=num_classes)
    model.apply(init_weights)  # Apply weight initialization
    train_model_with_early_stopping(model, train_loader, test_loader, num_epochs=30, learning_rate=0.0005)
else:
    print(f"File not found at path: {saved_data_path}")