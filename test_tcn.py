import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

# Training loop
def train_model(model, train_loader, test_loader, num_epochs=10, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training loop
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.float()  # Ensure the inputs are of type float
            labels = labels.long()   # Ensure labels are of type long

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Collect predictions and actual labels for accuracy tracking
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            if (i + 1) % 10 == 0:
                print(f"Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # Calculate training accuracy
        train_accuracy = accuracy_score(train_labels, train_preds)
        print(f"Epoch {epoch+1} - Training Loss: {train_loss/len(train_loader):.4f}, Training Accuracy: {train_accuracy:.4f}")

        # Validation loop
        model.eval()
        test_loss = 0.0
        test_preds = []
        test_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.float()
                labels = labels.long()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                # Collect predictions and actual labels
                preds = torch.argmax(outputs, dim=1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        # Calculate test accuracy
        test_accuracy = accuracy_score(test_labels, test_preds)
        print(f"Epoch {epoch+1} - Validation Loss: {test_loss/len(test_loader):.4f}, Validation Accuracy: {test_accuracy:.4f}\n")
        
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # print(f"Before Chomp1d: {x.shape}")
        x = x[:, :, :-self.chomp_size]
        # print(f"After Chomp1d: {x.shape}")
        return x

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # print(f"Input to TemporalBlock: {x.shape}")
        out = self.net(x)
        # print(f"Output of convs and activations (before residual connection): {out.shape}")
        if self.downsample is not None:
            res = self.downsample(x)
            # print(f"Reshaped residual (after downsampling): {res.shape}")
        else:
            res = x
        output = self.relu(out + res)
        # print(f"Final output after residual connection and ReLU: {output.shape}")
        return output

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        for i, layer in enumerate(self.network):
            # print(f"Passing input through layer {i+1}")
            x = layer(x)
        return x
    
# Define the TCN classifier for this task (using your existing TCN model)
class TCNClassifier(nn.Module):
    def __init__(self, input_size, num_channels, num_classes):
        super(TCNClassifier, self).__init__()
        # Use your existing TCN architecture
        self.tcn = TCN(input_size, num_channels)
        # Final classification layer
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        # Forward pass through the TCN layers
        output = self.tcn(x)
        # Take the last output from the sequence
        output = output[:, :, -1]
        # Pass it through the final linear layer for classification
        return self.fc(output)
    

    
class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index]    

def my_collate(batch):
    features, labels = zip(*batch)
    
    # Log the initial shapes of the features and labels
    # print("Initial shapes of features and labels:")
    for i, f in enumerate(features):
        print(f"Feature {i} shape: {f.shape}, Label: {labels[i]}")
    
    # Find the maximum number of channels in the batch (dim 1)
    max_channels = max(f.shape[1] for f in features)
    # print(f"Max number of channels in batch: {max_channels}")
    
    # Pad each feature tensor to match the max channels (dim 1)
    padded_features = []
    for i, f in enumerate(features):
        padded_feature = F.pad(f, (0, 0, 0, max_channels - f.shape[1]))  # Padding along channels (dim 1)
        # print(f"Padded Feature {i} shape: {padded_feature.shape}")  # Log the shape after padding
        padded_features.append(padded_feature)
    
    # Stack all the padded features and labels into tensors
    try:
        padded_features = torch.stack(padded_features)
        # print(f"Shape after stacking padded features: {padded_features.shape}")
    except Exception as e:
        print(f"Error during stacking: {e}")
        
        
    
    # Squeeze out the extra dimension at index 1 (the channel dimension)
    padded_features = padded_features.squeeze(1)
    # print(f"Shape after squeezing at index 1: {padded_features.shape}")
    
    padded_features = padded_features.permute(0, 2, 1)
    # print(f"Shape after permuting: {padded_features.shape}")
    
    # Convert labels to tensor
    labels = torch.tensor(labels)
    # print(f"Labels tensor shape: {labels.shape}")
    
    return padded_features, labels




# Ensure the correct file path
url = 'process_text_audio_seperatly\\text_features_only'
file_name = 'text_features_and_labels.pt'

# Construct the absolute path
saved_data_path = os.path.join(os.getcwd(), url, file_name)

class_labels = ['A1', 'A2', 'B1_1', 'B1_2', 'B2', 'C1', 'C2']
input_size = 1024  # Concatenated features
num_channels = [16, 32, 64]
num_classes = len(class_labels)


# Check if the file exists at the constructed path
if os.path.exists(saved_data_path):
    # Load the saved data
    loaded_data = torch.load(saved_data_path)

    # Extract features and labels
    features_list = loaded_data['features']
    labels = loaded_data['labels']
    # Print general statistics
    # print("Number of samples:", len(features_list))
    
    
    dataset = MyDataset(features_list, labels)
    # Split the dataset into training and validation sets
    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
          # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=my_collate)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=my_collate)
    # print(f"Train dataset size: {len(train_dataset)}")
    # print(f"Test dataset size: {len(test_dataset)}")
    
    # Example: Print the shape of the first few samples
    # for i in range(10):  # Change the range if you want to print more samples
    #     print(f"Sample {i+1} - Features shape: {features_list[i].shape}, Label: {labels[i]}")
    
    model = TCNClassifier(input_size=input_size, num_channels=num_channels, num_classes=num_classes)
    # Train the model
    train_model(model, train_loader, test_loader, num_epochs=10, learning_rate=0.001)


else:
    print(f"File not found at path: {saved_data_path}")
    
    
    
    