import os
import torch
from transformers import (
    XLNetTokenizer,
    XLNetForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import numpy as np
import whisper

# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Handle out-of-memory errors
def handle_oom_error(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('Out of memory error caught. Adjusting batch size or model parameters.')
                torch.cuda.empty_cache()
                # Implement logic to adjust batch size or model parameters
            else:
                raise e
    return wrapper

# Step 1: Load and transcribe audio files using Whisper
class AudioDataset(Dataset):
    def __init__(self, audio_dir, model_name='medium'):
        self.audio_dir = audio_dir
        self.audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        self.transcriptions = []
        self.labels = []
        self.model_name = model_name
        self.load_and_transcribe()

    def load_and_transcribe(self):
        transcriptions_file = f'transcriptions_{self.model_name}.pt'
        labels_file = 'labels.pt'
        if os.path.exists(transcriptions_file) and os.path.exists(labels_file):
            print('Loading saved transcriptions...')
            self.transcriptions = torch.load(transcriptions_file)
            self.labels = torch.load(labels_file)
        else:
            print('Transcribing audio files using Whisper...')
            # Load Whisper model
            whisper_model = whisper.load_model(self.model_name, device=device)

            for audio_file in tqdm(self.audio_files):
                audio_path = os.path.join(self.audio_dir, audio_file)
                # Transcribe audio using Whisper
                result = whisper_model.transcribe(audio_path)
                transcription = result['text']
                self.transcriptions.append(transcription)
                # Extract label from filename
                label = self.get_label_from_filename(audio_file)
                self.labels.append(label)
            # Save transcriptions and labels
            torch.save(self.transcriptions, transcriptions_file)
            torch.save(self.labels, labels_file)

    def get_label_from_filename(self, filename):
        # Implement logic to extract label from filename
        # For example, if filenames are like 'label_something.wav'
        label = filename.split('_')[0]
        return label

    def __len__(self):
        return len(self.transcriptions)

    def __getitem__(self, idx):
        return self.transcriptions[idx], self.labels[idx]

# Initialize the dataset
audio_dir = r'D:\audios'
dataset = AudioDataset(audio_dir, model_name='medium')

# Step 2: Prepare data, split into train and test sets
print('Splitting data into training and testing sets...')
dataset_size = len(dataset)
indices = list(range(dataset_size))
np.random.shuffle(indices)

train_split = int(np.floor(0.8 * dataset_size))
train_indices, test_indices = indices[:train_split], indices[train_split:]

train_transcriptions = [dataset[i][0] for i in train_indices]
train_labels = [dataset[i][1] for i in train_indices]
test_transcriptions = [dataset[i][0] for i in test_indices]
test_labels = [dataset[i][1] for i in test_indices]

# Save the indices to ensure consistency
torch.save(train_indices, 'train_indices.pt')
torch.save(test_indices, 'test_indices.pt')

# Step 3: Tokenize the text using XLNet tokenizer
tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')

# Create a custom dataset class for the text data
class TextDataset(Dataset):
    def __init__(self, transcriptions, labels, tokenizer, max_length=512):
        self.transcriptions = transcriptions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_id = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        self.encoded_inputs = self.tokenize_transcriptions()

    def tokenize_transcriptions(self):
        print('Tokenizing transcriptions...')
        encoded_inputs = self.tokenizer(
            self.transcriptions,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return encoded_inputs

    def __len__(self):
        return len(self.transcriptions)

    def __getitem__(self, idx):
        input_ids = self.encoded_inputs['input_ids'][idx]
        attention_mask = self.encoded_inputs['attention_mask'][idx]
        label = self.label_to_id[self.labels[idx]]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Prepare training and testing datasets
train_text_dataset = TextDataset(train_transcriptions, train_labels, tokenizer)
test_text_dataset = TextDataset(test_transcriptions, test_labels, tokenizer)

# Create dataloaders
batch_size = 2  # Adjust based on GPU memory
train_dataloader = DataLoader(train_text_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_text_dataset, batch_size=batch_size, shuffle=False)

# Step 4: Initialize XLNet model for sequence classification
model = XLNetForSequenceClassification.from_pretrained(
    'xlnet-large-cased',
    num_labels=len(train_text_dataset.label_to_id)
).to(device)

# Step 5: Fine-tune the model
optimizer = AdamW(model.parameters(), lr=1e-5)
epochs = 3
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

# Training function with OOM handling
def train(model, dataloader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc='Training'):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('Out of memory error during training. Skipping batch.')
                optimizer.zero_grad()
                torch.cuda.empty_cache()
            else:
                raise e
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Evaluation function
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            logits = outputs.logits
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_samples += labels.size(0)
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions.double() / total_samples
    return avg_loss, accuracy

# Training loop
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    train_loss = train(model, train_dataloader, optimizer, scheduler)
    print(f'Training loss: {train_loss}')
    val_loss, val_accuracy = evaluate(model, test_dataloader)
    print(f'Validation loss: {val_loss}, Validation accuracy: {val_accuracy}')

# Step 6: Save the trained model
output_dir = './xlnet_finetuned_model_whisper'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(f'Saving model to {output_dir}')
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
