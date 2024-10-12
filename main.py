import torch
from scipy.spatial.distance import euclidean
from tcn import TCNClassifier
from torch.utils.data import DataLoader, random_split
import torch
from padding import collate_fn, logger
from torch.nn.utils.rnn import pad_sequence
from extract_features import extract_features_labels
from IELTSDatasetClass import IELTSDataset
# Parameters
class_labels = ['A1', 'A2', 'B1_1', 'B1_2', 'B2', 'C1', 'C2']
input_size = 1024  # Concatenated features
num_channels = [16, 32, 64]
num_classes = len(class_labels)
batch_size = 16
epochs = 15
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
url = 'features_extracted'
print(device)
#STEP1: EXTRACT FEATURES AND LABELS FROM AUDIOS.
features_list, labels = extract_features_labels()


if len(features_list) != len(labels):
    logger.error(f"Inconsistent dataset sizes: features_list={len(features_list)}, labels={len(labels)}")
    raise ValueError("Mismatch between features and labels.")


# Step 2: Pad features to the length of the longest feature
# Each tensor in features_list is of shape [1, sequence_length, 1024]
# Remove the batch dimension (1) for padding
features_no_batch = [feature.squeeze(0) for feature in features_list]
# Pad features along the time dimension (dim=0) to match the longest sequence
padded_features = pad_sequence(features_no_batch, batch_first=True, padding_value=0.0)  # Shape: [batch_size, max_seq_len, 1024]
# STEP 2.1: USE MASKING
# Assume padded_features is of shape [batch_size, max_seq_len, feature_dim]
batch_size, max_seq_len, feature_dim = padded_features.size()
# Create a mask indicating where the real data is
mask = (padded_features.sum(dim=2) != 0).int()  # Shape: [batch_size, max_seq_len]
# If the sum of a feature vector along the feature dimension is zero, it's likely padding
print("Mask shape:", mask.shape)
print("Mask:", mask)  # 1 where there's data, 0 where there's padding
# Convert labels to a tensor for easier use
labels_tensor = torch.tensor(labels)
logger.info(f"padded features shape: {padded_features.shape}")
logger.info(f"labels tensor shape: {labels_tensor.shape}")

# Save features and labels to a file
data_to_save = {'features': features_list, 'labels': labels}
torch.save(data_to_save, f'{url}\\features_labels1.pt')

#STEP3: CREATE THE DATA SET, SPLIT IT TO TRAINING AND TESTING
# Create a dataset and split it into training and testing sets
dataset = IELTSDataset(padded_features, labels)
# Splitting the dataset into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
logger.debug(f"Dataset split into train size: {train_size}, test size: {test_size}")

# STEP4:
# Define batch size
batch_size = 16# Create DataLoaders for training and testing sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Optional: log DataLoader creation
logger.debug(f"Train loader and test loader created with batch size: {batch_size}")

# Initialize the model, loss function, and optimizer
model = TCNClassifier(input_size, num_channels, num_classes).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
logger.info(f"Model initialized with input size: {input_size}, learning rate: {learning_rate}")

# Training Loop
for epoch in range(epochs):
    logger.info(f"Starting epoch {epoch+1}/{epochs}")
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        logger.debug(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}]")

        # Extract features and labels from the batch
        batch_features = batch['feature'].to(device)  # Shape: [batch_size, sequence_length, input_size]
        batch_labels = batch['label'].to(device)      # Shape: [batch_size]
        logger.debug(f"Batch features shape: {batch_features.shape}, Batch labels shape: {batch_labels.shape}")

        # Create the mask
        mask = (batch_features.sum(dim=2) != 0).int()  # Shape: [batch_size, sequence_length]
        logger.debug(f"Mask shape: {mask.shape}")

        # Reshape features to [batch_size, input_size, sequence_length] as required by the model
        batch_features = batch_features.permute(0, 2, 1)
        logger.debug(f"Batch features after permutation shape: {batch_features.shape}")

        # Forward pass with mask
        try:
            # Forward pass through the model with the mask
            outputs = model(batch_features, mask=mask)
            logger.debug(f"Outputs shape after forward pass: {outputs.shape}")

            # Calculate loss
            loss = criterion(outputs, batch_labels)
            logger.debug(f"Loss for batch {batch_idx+1}: {loss.item()}")

        except Exception as e:
            logger.error(f"Error during forward pass: {e}")
            logger.debug(f"Mask shape: {mask.shape}, Batch features shape after permutation: {batch_features.shape}")
            continue

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Calculate and log average loss for the epoch
    avg_loss = total_loss / len(train_loader)
    logger.info(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")

# Save the trained model
model_path = "trained_tcn_model.pth"
torch.save(model.state_dict(), model_path)
logger.info(f"Model saved to {model_path}")

# Evaluation loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        logger.debug(f"Evaluating Batch [{batch_idx+1}], Feature Shape: {batch['feature'].shape}, Label Shape: {batch['label'].shape}")

        # Extract features and labels from the batch
        batch_features = batch['feature'].to(device)  # Shape: [batch_size, max_seq_len, input_size]
        batch_labels = batch['label'].to(device)      # Shape: [batch_size]
        logger.debug(f"Batch features shape: {batch_features.shape}, Batch labels shape: {batch_labels.shape}")

        # Create the mask
        mask = (batch_features.sum(dim=2) != 0).int()  # Shape: [batch_size, max_seq_len]
        logger.debug(f"Mask shape: {mask.shape}")

        # Reshape features to [batch_size, input_size, max_seq_len] as required by the model
        batch_features = batch_features.permute(0, 2, 1)
        logger.debug(f"Batch features after permutation shape: {batch_features.shape}")

        # Forward pass with mask
        try:
            # Forward pass through the model with the mask
            outputs = model(batch_features, mask=mask)
            logger.debug(f"Outputs shape after forward pass: {outputs.shape}")

            # Determine predicted labels
            _, predicted = torch.max(outputs, 1)
            logger.debug(f"Predicted labels: {predicted}, Batch labels: {batch_labels}")

            # Calculate accuracy
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            logger.debug(f"Mask shape: {mask.shape}, Batch features shape after permutation: {batch_features.shape}")
            continue

# Calculate and log accuracy
accuracy = correct / total if total > 0 else 0
logger.info(f"Test Accuracy: {accuracy * 100:.2f}%")