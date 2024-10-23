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
import logging
import colorlog
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s: %(message)s",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
))

logger = logging.getLogger('IELTSLogger')
logger.setLevel(logging.DEBUG)  # Adjust as needed
logger.addHandler(handler)


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
logger.debug(f'Using device: {device}')

# Step 3: Load data from JSON file
data_file = './transcriptions.jsonl'  # Replace with your JSON file path
with open(data_file, 'r', encoding='utf-8') as f:
    data = json.load(f)
    logger.info(f"Loaded {len(data)} samples from {data_file}")
logger.info(f"Loaded {len(data)} samples from {data_file}")
texts = []
labels = []
for item in data:
    texts.append(item['text'])
    labels.append(item['label'])

logger.debug(f'Total samples: {len(texts)}')

# Step 4: Prepare label mappings
label_set = sorted(set(labels))
label_to_id = {label: idx for idx, label in enumerate(label_set)}
id_to_label = {idx: label for label, idx in label_to_id.items()}
num_labels = len(label_set)
logger.debug(f'Number of labels: {num_labels}')
logger.debug(f'Label to ID mapping: {label_to_id}')

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

logger.debug(f'Training samples: {len(train_texts)}')
logger.debug(f'Validation samples: {len(val_texts)}')

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

# Step 8: Initialize tokenizer and datasets
tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')

train_dataset = TextDataset(train_texts, train_labels, tokenizer)
val_dataset = TextDataset(val_texts, val_labels, tokenizer)
logger.debug(f'Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}')


# Step 9: Create DataLoaders
batch_size = 2  # Adjust based on GPU memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Step 10: Initialize model
model = XLNetForSequenceClassification.from_pretrained('xlnet-large-cased', num_labels=num_labels)
model = model.to(device)

# Step 11: Set up optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

epochs = 3
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
            scheduler.step()
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_samples += labels.size(0)
            
            # Debugging statements
            logger.debug(f'Batch size: {labels.size(0)}, Loss: {loss.item()}')
        
        except RuntimeError as e:
            if 'out of memory' in str(e):
                logger.debug('Out of memory error during training. Moving batch to CPU.')
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                
                # Move batch to CPU and try processing again
                input_ids = input_ids.cpu()
                attention_mask = attention_mask.cpu()
                labels = labels.cpu()
                model.to("cpu")

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
                    scheduler.step()

                    total_loss += loss.item()
                    
                    preds = torch.argmax(logits, dim=1)
                    correct_predictions += torch.sum(preds == labels)
                    total_samples += labels.size(0)
                    
                    # Move model back to GPU
                    model.to(device)
                    
                    logger.debug(f'Recovered batch processed on CPU. Batch size: {labels.size(0)}, Loss: {loss.item()}')
                
                except RuntimeError as inner_e:
                    print(f'Failed to process batch on CPU. Error: {inner_e}')
                    continue  # Skip the batch and move to the next one

            else:
                raise e
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions.double() / total_samples
    logger.debug(f'Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
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
    logger.debug(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    return avg_loss, accuracy.item()

# Step 13: Training loop with model saving
for epoch in range(epochs):
    logger.debug(f'\nEpoch {epoch + 1}/{epochs}')
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
    val_loss, val_acc = eval_model(model, val_loader, device, epoch)
    
    # Save model after each epoch
    output_dir = f'./xlnet_finetuned_epoch_{epoch+1}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.debug(f'Model saved to {output_dir}')

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
logger.info('Applying t-SNE...')
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features)

# Step 16: Plot t-SNE visualization
plt.figure(figsize=(12, 8))
for label_id in np.unique(labels_array):
    indices = labels_array == label_id
    plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label=id_to_label[label_id], alpha=0.7)
plt.legend()
plt.title('t-SNE of XLNet Classification Outputs')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True)
plt.show()
