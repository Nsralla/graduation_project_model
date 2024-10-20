import torch
import torch.nn as nn
import os
from colorama import Fore, Style, init
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.model_selection import train_test_split
from padding import logger

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2  # Correct padding to maintain sequence length
        self.conv = nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()  # Add ReLU after each convolution (except the last one)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, apply_relu=True):
        # logger.info(Fore.CYAN + f"Before Conv: {x.shape}")
        x = self.conv(x)
        # logger.info(Fore.YELLOW + f"After Conv: {x.shape}")
        if apply_relu:  # Apply ReLU for the first two convolution layers, skip for the last one
            x = self.relu(x)
            logger.info(Fore.GREEN + f"After ReLU: {x.shape}")
        x = self.dropout(x)
        # logger.info(Fore.MAGENTA + f"After Dropout: {x.shape}")
        return x

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, num_classes, kernel_size=2, dropout=0.2):
        super().__init__()

        # First Conv -> Pool
        self.conv1 = TemporalBlock(num_inputs, num_channels[0], kernel_size, dilation=1, dropout=dropout)
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)

        # Second Conv -> Pool
        self.conv2 = TemporalBlock(num_channels[0], num_channels[1], kernel_size, dilation=1, dropout=dropout)
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)

        # Third Conv (No ReLU after this one, as per instructions)
        self.conv3 = TemporalBlock(num_channels[1], num_channels[2], kernel_size, dilation=1, dropout=dropout)

        # Final fully connected layer
        self.fc = nn.Linear(num_channels[2], num_classes)

    def forward(self, x):
        # Conv -> Pool -> ReLU
        # logger.info(Fore.BLUE + "Passing through Conv1 and Pool1")
        x = self.conv1(x)
        x = self.pool1(x)
        # logger.info(Fore.CYAN + f"After Pool1: {x.shape}")

        # Conv -> Pool -> ReLU
        # logger.info(Fore.BLUE + "Passing through Conv2 and Pool2")
        x = self.conv2(x)
        x = self.pool2(x)
        # logger.info(Fore.CYAN + f"After Pool2: {x.shape}")

        # Conv (No ReLU after the last conv)
        # logger.info(Fore.BLUE + "Passing through Conv3 (No ReLU)")
        x = self.conv3(x, apply_relu=False)
        # logger.info(Fore.CYAN + f"After Conv3: {x.shape}")

        # Global pooling or mean to collapse sequence dimension
        # logger.info(Fore.BLUE + "Applying Global Mean Pooling")
        x = torch.mean(x, dim=-1)  # Collapse the sequence dimension
        # logger.info(Fore.CYAN + f"After Global Mean Pooling: {x.shape}")

        # Fully connected layer to classify into classes
        x = self.fc(x)
        # logger.info(Fore.GREEN + f"After Fully Connected Layer (Output): {x.shape}")
        return x  # Output raw logits for CrossEntropyLoss (no softmax needed)
class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Function to initialize weights
def init_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)

# Custom collate function for padding
def my_collate(batch):
    features, labels = zip(*batch)
    max_channels = max(f.shape[1] for f in features)
    padded_features = [F.pad(f, (0, 0, 0, max_channels - f.shape[1])) for f in features]
    padded_features = torch.stack(padded_features)
    padded_features = padded_features.squeeze(1).permute(0, 2, 1)
    labels = torch.tensor(labels)
    return padded_features, labels
# Load the data
url = 'process_text_audio_seperatly\\text_features_only'
file_name = 'text_features_and_labels.pt'
saved_data_path = os.path.join(os.getcwd(), url, file_name)

class_labels = ['A1', 'A2', 'B1_1','B1_2','B2', 'C1', 'C2']
num_inputs = 1024
num_channels = [32, 64, 128]  # Increased channels for better capacity
num_classes = len(class_labels)

# Check if the file exists
if os.path.exists(saved_data_path):
    loaded_data = torch.load(saved_data_path)
    features_list = loaded_data['features']
    labels = loaded_data['labels']

    # Filter the dataset to keep target labels (C1, C2, A1, A2, B2)
    target_labels = [class_labels.index('C1'), class_labels.index('C2'), class_labels.index('A1'), class_labels.index('A2'), class_labels.index('B2')]

    # Create a new dataset
    dataset = MyDataset(features_list, labels)

    # Split the dataset into training and validation sets (80% training, 20% validation)
    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=my_collate)

    # Initialize the model
    model = TCN(num_inputs=num_inputs, num_channels=num_channels, num_classes=num_classes)

    model.apply(init_weights)  # Apply weight initialization

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Training loop
    num_epochs = 30
    for epoch in range(num_epochs):
        # Training phase
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct_predictions = 0
        total_train = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Clear previous gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update model parameters

            running_loss += loss.item() * inputs.size(0)  # Accumulate the loss

            # Calculate training accuracy
            _, predicted = torch.max(outputs, dim=1)
            correct_predictions += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / total_train
        train_accuracy = correct_predictions / total_train

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct_predictions_val = 0
        total_val = 0

        with torch.no_grad():  # Disable gradient computation for validation
            for inputs, labels in val_loader:
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Compute validation loss
                val_loss += loss.item() * inputs.size(0)  # Accumulate validation loss

                # Calculate validation accuracy
                _, predicted = torch.max(outputs, dim=1)
                correct_predictions_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_loss = val_loss / total_val
        val_accuracy = correct_predictions_val / total_val

        # Print results for the epoch
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

else:
    print(f"File not found at path: {saved_data_path}")

# Ensure the correct file path
# url = 'intermediate_results\\processed_features'
# file_name = 'feature_0.pt'
# saved_data_path = os.path.join(os.getcwd(), url, file_name)
# if os.path.exists(saved_data_path):
#     # Load the saved data
#     loaded_data = torch.load(saved_data_path)
#     # Extract features and labels
#     feature = loaded_data['pooled_feature']
#     label = loaded_data['label']
#     print(feature.shape)
#     print(label)
#     feature = feature.permute(0,2,1)
# # Sample inputs
# classes = ['A1','A2','B1_1','B1_2','B2','C1','C2']
# num_classes = 7  # A1, A2, B1_1, B1_2, B2, C1, C2
# batch_size = feature.shape[0]
# num_channels = feature.shape[1]
# sequence_length = feature.shape[2]
# # Initialize the TCN model
# tcn = TCN(num_inputs=num_channels, num_channels=[32, 64, 128], num_classes=num_classes,kernel_size=3, dropout=0.2)
# # Create a random input tensor
# x = torch.randn(batch_size, num_channels, sequence_length)
# # Pass it through the model
# output = tcn(x)
# print(output.shape)  # Output will have shape (batch_size, 64, sequence_length)
# print(output)
# probabilities = torch.softmax(output, dim=-1)
# print(probabilities)
# predicted_class = torch.argmax(probabilities, dim=-1)
# print(f"Predicted Class Index: {predicted_class.item()}")
# class_names = ['A1', 'A2', 'B1_1', 'B1_2', 'B2', 'C1', 'C2']
# predicted_label = class_names[predicted_class.item()]
# print(f"Predicted Proficiency Level: {predicted_label}")
# criterion = nn.CrossEntropyLoss()
# target_label = torch.tensor([2])  # Example target class, say B1_1
# # Compute loss
# loss = criterion(output, target_label)
# print("loss: ",loss)

