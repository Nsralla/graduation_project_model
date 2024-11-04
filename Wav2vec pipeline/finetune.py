# Install necessary packages
!pip install transformers torchaudio scikit-learn

# Install ffmpeg for torchaudio to handle mp3 files
!apt-get install -y ffmpeg

# Import statements
import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torchaudio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import pickle
import torch
import gc

# Ensure matplotlib plots inline
%matplotlib inline

# Step 1: Set up random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Step 2: Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Step 3: Define the path to the audio files and preprocessed data
audio_dir = '/content/drive/MyDrive/audios/audios'  # Update based on your directory in Colab
processed_data_dir = '/content/processed_audios'  # Directory to save preprocessed audio

# Create directories if they don't exist
os.makedirs(audio_dir, exist_ok=True)
os.makedirs(processed_data_dir, exist_ok=True)

# Step 4: Extract labels from filenames
def extract_label_from_filename(filename):
    filename = filename.lower()
    possible_labels = ['a1', 'a2', 'b1_1', 'b1_2', 'b2', 'c1', 'c2']

    # Check if any known label is in the filename
    for label in possible_labels:
        if label in filename:
            return label

    # If no label is found, return None
    print(f"No label found in filename: {filename}")
    return None

# Collect all audio files and their labels
data = []
for root, dirs, files in os.walk(audio_dir):
    for file in files:
        if file.endswith('.mp3') or file.endswith('.wav'):
            label = extract_label_from_filename(file)
            if label is not None:
                data.append({
                    'audio_path': os.path.join(root, file),
                    'label': label
                })
            else:
                print(f"Skipping file {file} as no label was found.")

print(f'Total samples found: {len(data)}')

# Step 5: Prepare label mappings
label_set = sorted(set(item['label'] for item in data))
label_to_id = {label: idx for idx, label in enumerate(label_set)}
id_to_label = {idx: label for label, idx in label_to_id.items()}
num_labels = len(label_set)
print(f'Number of labels: {num_labels}')
print(f'Label to ID mapping: {label_to_id}')

# Encode labels
for item in data:
    item['label_id'] = label_to_id[item['label']]

# Step 6: Preprocess audio files and save them
def preprocess_and_save_audio(data, processed_data_dir):
    print('Preprocessing audio files...')
    processed_data = []
    for item in tqdm(data):
        audio_path = item['audio_path']
        try:
            audio_input, sample_rate = torchaudio.load(audio_path)

            # Resample if necessary
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                audio_input = resampler(audio_input)
                sample_rate = 16000

            # Convert to mono if necessary
            if audio_input.shape[0] > 1:
                audio_input = torch.mean(audio_input, dim=0, keepdim=True)

            # Remove VAD step
            # Previously, VAD was applied here

            # Save processed audio to file
            processed_audio_path = os.path.join(
                processed_data_dir,
                os.path.splitext(os.path.basename(item['audio_path']))[0] + '.pt'
            )
            torch.save(audio_input, processed_audio_path)

            processed_data.append({
                'audio_path': processed_audio_path,
                'label_id': item['label_id']
            })
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
    return processed_data

# Preprocess and save audio files
processed_data = preprocess_and_save_audio(data, processed_data_dir)

# Step 7: Split data into training and validation sets
data_size = len(processed_data)
indices = list(range(data_size))
random.shuffle(indices)

train_split = int(0.8 * data_size)
train_indices = indices[:train_split]
val_indices = indices[train_split:]

train_data = [processed_data[i] for i in train_indices]
val_data = [processed_data[i] for i in val_indices]

print(f'Training samples: {len(train_data)}')
print(f'Validation samples: {len(val_data)}')

# Step 8: Define custom Dataset class for preprocessed audio data
class PreprocessedAudioDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio_input = torch.load(item['audio_path']).squeeze()

        # Get the input values from the processor
        input_values = self.processor(audio_input.numpy(), sampling_rate=16000).input_values[0]

        label = item['label_id']
        return {
            'input_values': torch.tensor(input_values, dtype=torch.float),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Step 9: Define data collator
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features):
        input_values = [f["input_values"] for f in features]
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)

        batch = self.processor.pad(
            {"input_values": input_values},
            padding=self.padding,
            return_tensors="pt"
        )
        batch["labels"] = labels
        return batch

# Step 10: Initialize processor
from transformers import Wav2Vec2Config

processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')

# Step 11: Initialize model
config = Wav2Vec2Config.from_pretrained('facebook/wav2vec2-base', num_labels=num_labels)
model = Wav2Vec2ForSequenceClassification.from_pretrained('facebook/wav2vec2-base', config=config)
model.freeze_feature_extractor()  # Freeze feature extractor layers
model = model.to(device)

# Freeze Lower Transformer Layers
# Freezing the first 8 transformer layers
for layer in model.wav2vec2.encoder.layers[:8]:
    for param in layer.parameters():
        param.requires_grad = False

# Print parameter names and their trainable status
for name, param in model.named_parameters():
    print(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")

# Initialize GradScaler
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# Step 12: Initialize datasets and data loaders
train_dataset = PreprocessedAudioDataset(train_data, processor)
val_dataset = PreprocessedAudioDataset(val_data, processor)

batch_size = 10  # Adjust batch size as needed
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator, num_workers=4)

# Step 13: Set up optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

epochs = 30
total_steps = len(train_loader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=total_steps)

# Step 14: Define training and evaluation functions
def train_epoch(model, data_loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(data_loader, desc=f'Training Epoch {epoch+1}')
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Move batch to the device
            input_values = batch['input_values'].to(device)
            labels = batch['labels'].to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Forward and backward pass
            with autocast():
                outputs = model(input_values=input_values, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

            scaler.scale(loss).backward()  # Backpropagation with mixed precision
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

            # Calculate accuracy
            preds = torch.argmax(logits, dim=-1)
            correct_predictions += torch.sum(preds == labels)
            total_samples += labels.size(0)

        except torch.cuda.OutOfMemoryError:
            print("OOM error encountered, skipping this batch.")
            torch.cuda.empty_cache()
            gc.collect()
            continue

        # Update progress bar description
        running_loss = total_loss / (batch_idx + 1)
        running_accuracy = correct_predictions.double() / total_samples
        progress_bar.set_postfix({
            'Loss': f'{running_loss:.4f}',
            'Acc': f'{running_accuracy:.4f}'
        })

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions.double() / total_samples
    print(f'\nTraining Epoch {epoch+1} Summary: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    return avg_loss, accuracy.item()

def eval_model(model, data_loader, device, epoch):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(data_loader, desc=f'Validating Epoch {epoch+1}')
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            try:
                input_values = batch['input_values'].to(device)
                labels = batch['labels'].to(device)

                with autocast():
                    outputs = model(
                        input_values=input_values,
                        labels=labels
                    )
                    loss = outputs.loss
                    logits = outputs.logits

                total_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                correct_predictions += torch.sum(preds == labels)
                total_samples += labels.size(0)

            except torch.cuda.OutOfMemoryError:
                print("OOM error encountered during validation, skipping this batch.")
                torch.cuda.empty_cache()
                gc.collect()
                continue

            # Update progress bar with running loss and accuracy
            running_loss = total_loss / (batch_idx + 1)
            running_accuracy = correct_predictions.double() / total_samples
            progress_bar.set_postfix({
                'Loss': f'{running_loss:.4f}',
                'Acc': f'{running_accuracy:.4f}'
            })

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions.double() / total_samples
    print(f'\nValidation Epoch {epoch+1} Summary: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    return avg_loss, accuracy.item()

from google.colab import files
import shutil

# Step 15: Training loop with model saving and logging
# Training loop with model saving and logging
for epoch in range(epochs):
    print(f'\nEpoch {epoch + 1}/{epochs}')

    # Run training and evaluation
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
    val_loss, val_acc = eval_model(model, val_loader, device, epoch)

    # Save model, processor, optimizer, and scheduler as a checkpoint
    checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pt"
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc
    }, checkpoint_path)

    # Save the processor in a separate directory
    processor_dir = f"{checkpoint_dir}/processor_epoch_{epoch+1}"
    processor.save_pretrained(processor_dir)

    print(f"Checkpoint saved to {checkpoint_path}")
    print(f"Processor saved to {processor_dir}")

print("Training complete and all checkpoints saved.")

# Step 16: Extract features for t-SNE visualization
def get_model_outputs(model, data_loader, device):
    model.eval()
    features = []
    labels_list = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Extracting Features'):
            input_values = batch['input_values'].to(device)
            labels_batch = batch['labels']

            with autocast():
                outputs = model(
                    input_values=input_values,
                    labels=None
                )
                logits = outputs.logits  # Shape: [batch_size, num_labels]

            features.append(logits.cpu().numpy())
            labels_list.extend(labels_batch.numpy())

    features = np.concatenate(features, axis=0)
    labels_array = np.array(labels_list)
    return features, labels_array

# Get features and labels from validation set
features, labels_array = get_model_outputs(model, val_loader, device)

# Step 17: Apply t-SNE
print('Applying t-SNE...')
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features)

# Step 18: Plot t-SNE visualization
plt.figure(figsize=(12, 8))
for label_id in np.unique(labels_array):
    indices = labels_array == label_id
    plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label=id_to_label[label_id], alpha=0.7)
plt.legend()
plt.title('t-SNE of Fine-tuned Wav2Vec2 Classification Outputs')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True)
plt.show()
