# Import statements
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
    Wav2Vec2Config,
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchaudio
from dataclasses import dataclass
from typing import Union
import gc

# Step 1: Set up random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Step 2: Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Step 3: Define the path to the audio files
audio_dir = r'/content/drive/MyDrive/Youtube/Audios/training youtube'  # Update based on your directory

# Step 4: Extract labels from filenames
def extract_label_from_filename(filename):
    filename = filename.lower()
    possible_labels = ['a1', 'c1', 'c2']

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

# Step 6: Preprocess and save audio files to disk
def preprocess_audio(data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    processed_data = []
    print('Preprocessing audio files...')
    for item in tqdm(data):
        audio_path = item['audio_path']
        label = item['label_id']
        
        try:
            audio_input, sample_rate = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                audio_input = resampler(audio_input)
            
            # Convert to mono if necessary
            if audio_input.shape[0] > 1:
                audio_input = torch.mean(audio_input, dim=0, keepdim=True)
    
            audio_input = audio_input.squeeze()
            processed_audio_path = os.path.join(
                output_dir, os.path.basename(audio_path).replace('.mp3', '.pt').replace('.wav', '.pt')
            )
            torch.save({'audio': audio_input, 'label': label}, processed_audio_path)
            processed_data.append({'audio_path': processed_audio_path, 'label_id': label})
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
    return processed_data

# Preprocess audio files and save
processed_data_dir = './processed_audios'
processed_data = preprocess_audio(data, processed_data_dir)

# Step 7: Define custom Dataset class that loads preprocessed data
class PreprocessedAudioDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio_data = torch.load(item['audio_path'])
        audio_input = audio_data['audio']
        label = audio_data['label']

        # Get the input values from the processor
        input_values = self.processor(audio_input.numpy(), sampling_rate=16000).input_values[0]

        return {
            'input_values': torch.tensor(input_values, dtype=torch.float),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Step 8: Define data collator
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

# Step 9: Initialize processor
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')

# Step 10: Initialize model
config = Wav2Vec2Config.from_pretrained('facebook/wav2vec2-base', num_labels=num_labels)
model = Wav2Vec2ForSequenceClassification.from_pretrained('facebook/wav2vec2-base', config=config)
model.freeze_feature_extractor()  # Freeze feature extractor layers
model = model.to(device)

# Freeze more layers in the transformer encoder to speed up training
# For Wav2Vec2-base with 12 transformer layers, freeze first 10 layers
for layer in model.wav2vec2.encoder.layers[:8]:
    for param in layer.parameters():
        param.requires_grad = False

# Print parameter names and their trainable status
print("\nModel Parameters:")
for name, param in model.named_parameters():
    print(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")

# Step 11: Initialize dataset and data loader
train_dataset = PreprocessedAudioDataset(processed_data, processor)

batch_size = 6  # Adjust batch size based on your system's memory
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data_collator,
    pin_memory=True
)

# Step 12: Set up optimizer and scheduler
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=0.01)

epochs = 13  # Adjust the number of epochs as needed
total_steps = len(train_loader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
)

# Step 13: Initialize GradScaler for mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# Step 14: Define training function
def train_epoch(model, data_loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(data_loader, desc=f'Training Epoch {epoch+1}')
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to the device
        input_values = batch['input_values'].to(device)
        labels = batch['labels'].to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass with mixed precision
        with autocast():
            outputs = model(input_values=input_values, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

        # Calculate accuracy
        preds = torch.argmax(logits, dim=-1)
        correct_predictions += torch.sum(preds == labels)
        total_samples += labels.size(0)

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

# Step 15: Training loop with model saving
checkpoint_dir = './wav2vec2_finetuned_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

for epoch in range(epochs):
    print(f'\nEpoch {epoch + 1}/{epochs}')

    # Run training
    train_loss, train_acc = train_epoch(
        model, train_loader, optimizer, scheduler, device, epoch
    )

    # Save model and processor
    output_dir = f'{checkpoint_dir}/epoch_{epoch+1}'
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Model and processor saved to {output_dir}")

print("Training complete and all checkpoints saved.")

# Optional: t-SNE Visualization using training data
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
                    output_hidden_states=True,
                    return_dict=True
                )
                # Use the hidden states from the last layer
                hidden_states = outputs.hidden_states[-1]
                pooled_output = torch.mean(hidden_states, dim=1).cpu().numpy()  # Average pooling
            features.append(pooled_output)
            labels_list.extend(labels_batch.numpy())

    features = np.concatenate(features, axis=0)
    labels_array = np.array(labels_list)
    return features, labels_array

# Get features and labels from training set
features, labels_array = get_model_outputs(model, train_loader, device)

# Apply t-SNE
print('Applying t-SNE...')
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features)

# Plot t-SNE visualization
plt.figure(figsize=(12, 8))
for label_id in np.unique(labels_array):
    indices = labels_array == label_id
    plt.scatter(
        features_2d[indices, 0],
        features_2d[indices, 1],
        label=id_to_label[label_id],
        alpha=0.7
    )
plt.legend()
plt.title('t-SNE of Wav2Vec2 Embeddings on Training Data')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True)
plt.show()
