import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from transformers import XLNetTokenizer, XLNetForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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

# Step 3: Load data from JSON file
data_file = './transcriptions.jsonl'  # Replace with your JSON file path
data = []

with open(data_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

texts = []
labels = []
for item in data:
    texts.append(item['text'])
    labels.append(item['label'])

print(f'Total samples: {len(texts)}')

# Step 4: Prepare label mappings
label_set = sorted(set(labels))
label_to_id = {label: idx for idx, label in enumerate(label_set)}
id_to_label = {idx: label for label, idx in label_to_id.items()}
num_labels = len(label_set)
print(f'Number of labels: {num_labels}')
print(f'Label to ID mapping: {label_to_id}')

# Encode labels
encoded_labels = [label_to_id[label] for label in labels]

# Step 5: Split data into training and validation sets
data_size = len(texts)
indices = list(range(data_size))
random.shuffle(indices)

train_split = int(0.8 * data_size)
train_indices = indices[:train_split]
val_indices = indices[train_split:]

train_texts = [texts[i] for i in train_indices]
train_labels = [encoded_labels[i] for i in train_indices]
val_texts = [texts[i] for i in val_indices]
val_labels = [encoded_labels[i] for i in val_indices]

print(f'Training samples: {len(train_texts)}')
print(f'Validation samples: {len(val_texts)}')

# Step 6: Define custom Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            truncation=True,
            padding=False,  # Padding will be handled in the collate_fn
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # Shape: [seq_len]
            'attention_mask': encoding['attention_mask'].squeeze(0),  # Shape: [seq_len]
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Step 7: Define custom collate function
def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
    
    # Pad sequences
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks_padded = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
    
    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded,
        'labels': labels
    }

from transformers import BertTokenizer, BertForSequenceClassification

# Step 8: Initialize tokenizer and datasets
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

train_dataset = TextDataset(train_texts, train_labels, tokenizer)
val_dataset = TextDataset(val_texts, val_labels, tokenizer)

# Step 9: Create DataLoaders
batch_size = 12  # Adjust based on GPU memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
from transformers import BertConfig, BertForSequenceClassification
# Step 10: Initialize model
config = BertConfig.from_pretrained('bert-base-cased', num_labels=num_labels, hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.2)
model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=num_labels)
model = model.to(device)

# Freeze the entire model first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last two layers (classification layer and the last encoder layer)
def unfreeze_last_layers(model, num_layers=2):
    layers_to_unfreeze = list(model.children())[-num_layers:]  # Get the last `num_layers` layers
    for layer in layers_to_unfreeze:
        for param in layer.parameters():
            param.requires_grad = True

unfreeze_last_layers(model, num_layers=4)

# Step 11: Set up optimizer and scheduler (only unfrozen parameters will be optimized)
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5, eps=1e-8, weight_decay=0.01)

epochs = 15
total_steps = len(train_loader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Step 12: Define training and evaluation functions with OOM handling
def train_epoch(model, data_loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    for batch in tqdm(data_loader, desc=f'Training Epoch {epoch+1}'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()  # Update the learning rate scheduler here
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_samples += labels.size(0)
            
            # Debugging statements
            print(f'Batch size: {labels.size(0)}, Loss: {loss.item()}')
        
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('Out of memory error during training. Skipping batch.')
                optimizer.zero_grad()
                torch.cuda.empty_cache()
            else:
                raise e
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions.double() / total_samples
    print(f'Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    return avg_loss, accuracy.item()

def eval_model(model, data_loader, device, epoch):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f'Evaluating Epoch {epoch+1}'):
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
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions.double() / total_samples
    print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    return avg_loss, accuracy.item()

# Step 13: Training loop with model saving
for epoch in range(epochs):
    print(f'\nEpoch {epoch + 1}/{epochs}')
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
    val_loss, val_acc = eval_model(model, val_loader, device, epoch)
    
    # Save model after each epoch
    output_dir = f'./xlnet_finetuned_epoch_freezing_firstLayers_{epoch+1}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f'Model saved to {output_dir}')

# Step 14: Extract features for t-SNE visualization
def get_model_outputs(model, data_loader, device):
    model.eval()
    features = []
    labels_list = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Extracting Features'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['labels']
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
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

# Step 15: Apply t-SNE
print('Applying t-SNE...')
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features)

# Step 16: Plot t-SNE visualization
plt.figure(figsize=(12, 8))
for label_id in np.unique(labels_array):
    indices = labels_array == label_id
    plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label=id_to_label[label_id], alpha=0.7)
plt.legend()
plt.title('t-SNE of Base bert Classification Outputs')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True)
plt.show()
