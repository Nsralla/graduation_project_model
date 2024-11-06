import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report

# Step 1: Set up random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Step 2: Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Step 3: Load data from JSON file
data_file = './Icnale_training_transcription.jsonl'
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

# Step 5: Use all data as training data
train_texts = texts
train_labels = encoded_labels

print(f'Training samples: {len(train_texts)}')

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
        text = str(self.texts[idx])
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
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks_padded = torch.nn.utils.rnn.pad_sequence(
        attention_masks, batch_first=True, padding_value=0
    )

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded,
        'labels': labels
    }

# Step 8: Initialize tokenizer and dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

train_dataset = TextDataset(train_texts, train_labels, tokenizer)

# Step 9: Create DataLoader
batch_size = 16  # Adjust based on GPU memory
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)

# Step 10: Initialize model
config = BertConfig.from_pretrained(
    'bert-base-cased',
    num_labels=num_labels,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
)
model = BertForSequenceClassification.from_pretrained(
    'bert-base-cased', config=config
)
model = model.to(device)

# Unfreeze the entire model for fine-tuning
for param in model.parameters():
    param.requires_grad = True

# Step 11: Set up optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)

epochs = 25  # Adjust as needed
total_steps = len(train_loader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

# Step 12: Define training function
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

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions.double() / total_samples
    print(f'Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    return avg_loss, accuracy.item()

# Step 13: Training loop with model saving
for epoch in range(epochs):
    print(f'\nEpoch {epoch + 1}/{epochs}')
    train_loss, train_acc = train_epoch(
        model, train_loader, optimizer, scheduler, device, epoch
    )

    # Save model and tokenizer after each epoch
    output_dir = f'./bert_finetuned_epoch_{epoch + 1}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save optimizer and scheduler states
    checkpoint = {
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch + 1
    }
    checkpoint_path = os.path.join(output_dir, 'training_args.bin')
    torch.save(checkpoint, checkpoint_path)

    print(f'Model and tokenizer saved to {output_dir}')

# Step 14: Evaluation on training data
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(train_loader, desc='Evaluating on Training Data'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print('Classification Report on Training Data:')
print(classification_report(
    all_labels,
    all_preds,
    target_names=[id_to_label[i] for i in range(num_labels)],
    digits=4
))

# Step 15: t-SNE Visualization using training data
def get_model_outputs(model, data_loader, device):
    model.eval()
    features = []
    labels_list = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['labels']

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

            # Use the CLS token embeddings from the last hidden state
            hidden_states = outputs.hidden_states[-1]
            cls_embeddings = hidden_states[:, 0, :].cpu().numpy()  # [batch_size, hidden_size]
            features.append(cls_embeddings)
            labels_list.extend(labels_batch.numpy())

    features = np.concatenate(features, axis=0)
    labels_array = np.array(labels_list)
    return features, labels_array

# Get features and labels from training set
features, labels_array = get_model_outputs(model, train_loader, device)

# Apply t-SNE
print('Applying t-SNE...')
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
plt.title('t-SNE of BERT Embeddings on Training Data')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True)
plt.show()
