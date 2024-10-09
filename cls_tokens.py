import torch
from IELTSDataSet2 import IELTSDataSetForCLS
from torch.utils.data import random_split, DataLoader
from padding import logger
from cnn import FullyConnectedClassifier

# Define class labels and hyperparameters
class_labels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
input_size = 2048  # Concatenated features
num_hidden_units = 512  # Number of hidden units in the hidden layer
num_classes = len(class_labels)
batch_size = 16
epochs = 15
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging the configuration
logger.info(f"Starting training with the following configuration: "
            f"Input size: {input_size}, "
            f"Hidden units: {num_hidden_units}, "
            f"Classes: {num_classes}, "
            f"Batch size: {batch_size}, "
            f"Epochs: {epochs}, "
            f"Learning rate: {learning_rate}, "
            f"Device: {device}")

# STEP 1: LOAD THE DATA
logger.info("Loading the data...")
all_concatenated_cls_tokens = torch.load("all_cls_tokens_concatenated.pt")
logger.info(f"Data loaded successfully. Total samples: {len(all_concatenated_cls_tokens)}")

# Create dataset
dataset = IELTSDataSetForCLS(all_concatenated_cls_tokens)
logger.info("Dataset created successfully.")

# STEP 3: SPLIT DATA INTO TRAINING AND TESTING
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
training_data, testing_data = random_split(dataset, [train_size, test_size])
logger.info(f"Dataset split into training and testing sets: Training size = {train_size}, Testing size = {test_size}")

# STEP 4: CREATE DATA LOADERS
train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testing_data, batch_size=batch_size, shuffle=False)
logger.info("Data loaders created.")

# STEP 5: Initialize the model, loss function, and optimizer
model = FullyConnectedClassifier(input_size, num_hidden_units, num_classes).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
logger.info(f"Model initialized on {device}. Ready to start training.")

# Training + Evaluation loop
for epoch in range(epochs):
    # Training Phase
    logger.info(f"Starting epoch {epoch+1}/{epochs}")
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        logger.debug(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}]")

        # Unpack the batch into features and labels
        batch_features, batch_labels = batch
        batch_features = batch_features.to(device)  # Move features to device
        batch_labels = batch_labels.to(device)      # Move labels to device
        logger.debug(f"Batch features shape: {batch_features.shape}, Batch labels shape: {batch_labels.shape}")

        # Forward pass
        outputs = model(batch_features)  # Outputs will have shape [batch_size, num_classes]
        outputs = torch.squeeze(outputs, 1)  # Remove the extra dimension to get shape [batch_size, num_classes]
        logger.debug(f"Outputs shape after forward pass: {outputs.shape}")

        # Calculate loss
        loss = criterion(outputs, batch_labels)  # Outputs is [batch_size, num_classes], batch_labels is [batch_size]
        logger.debug(f"Loss for batch {batch_idx+1}: {loss.item()}")

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Calculate and log average loss for the epoch
    avg_loss = total_loss / len(train_loader)
    logger.info(f"Epoch [{epoch+1}/{epochs}] completed. Average Loss: {avg_loss:.4f}")

    # Evaluation Phase
    model.eval()  # Set the model to evaluation mode
    total_test_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient calculation
        for batch_idx, batch in enumerate(test_loader):
            batch_features, batch_labels = batch
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            # Forward pass for evaluation
            outputs = model(batch_features)
            outputs = torch.squeeze(outputs, 1)  # Ensure correct shape

            # Calculate loss for the test set
            loss = criterion(outputs, batch_labels)
            total_test_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
            correct_predictions += (predicted == batch_labels).sum().item()
            total_samples += batch_labels.size(0)

    # Calculate average test loss and accuracy
    avg_test_loss = total_test_loss / len(test_loader)
    accuracy = correct_predictions / total_samples
    logger.info(f"Evaluation completed. Average Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.4f}")

logger.info("Training complete.")
