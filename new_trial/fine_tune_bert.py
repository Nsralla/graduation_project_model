import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, AdamW, DistilBertConfig
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import logging
import colorlog
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
import optuna

# Set up logging
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

logger = logging.getLogger('ModelLogger')
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

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
logger.debug(f'Using device: {device}')

# Load data from JSON Lines file
data_file = './transcriptions.jsonl'
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

logger.debug(f'Total samples: {len(texts)}')

# Prepare label mappings
label_set = sorted(set(labels))
label_to_id = {label: idx for idx, label in enumerate(label_set)}
id_to_label = {idx: label for label, idx in label_to_id.items()}
num_labels = len(label_set)
logger.debug(f'Number of labels: {num_labels}')
logger.debug(f'Label to ID mapping: {label_to_id}')

# Encode labels
encoded_labels = [label_to_id[label] for label in labels]

# Define custom Dataset class
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

# Define custom collate function
def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)

    # Pad sequences
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks_padded = torch.nn.utils.rnn.pad_sequence(
        attention_masks, batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded,
        'labels': labels
    }

# Initialize tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

# Hyperparameter Optimization with Optuna
def objective(trial):
    # Hyperparameters to tune
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-1)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-4)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])

    # Update seed for each trial
    set_seed(42)

    # Adjust the dataset and dataloader
    max_length = 512
    dataset = TextDataset(texts, encoded_labels, tokenizer, max_length=max_length)

    # Implement Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_index, val_index) in enumerate(skf.split(texts, encoded_labels)):
        logger.info(f'Fold {fold + 1}')
        train_texts_fold = [texts[i] for i in train_index]
        val_texts_fold = [texts[i] for i in val_index]
        train_labels_fold = [encoded_labels[i] for i in train_index]
        val_labels_fold = [encoded_labels[i] for i in val_index]

        train_dataset = TextDataset(train_texts_fold, train_labels_fold, tokenizer, max_length=max_length)
        val_dataset = TextDataset(val_texts_fold, val_labels_fold, tokenizer, max_length=max_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        # Initialize model with dropout
        from transformers import DistilBertConfig

        config = DistilBertConfig.from_pretrained(
            'distilbert-base-cased',
            num_labels=num_labels,
            dropout=dropout_rate,
            attention_dropout=dropout_rate
        )
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-cased',
            config=config
        )
        model = model.to(device)

        # Set up optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        epochs = 50
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        # Training loop
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            correct_predictions = 0
            total_samples = 0

            for batch in train_loader:
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

            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = correct_predictions.double() / total_samples

            # Validation
            model.eval()
            total_loss = 0
            correct_predictions = 0
            total_samples = 0

            with torch.no_grad():
                for batch in val_loader:
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

            avg_val_loss = total_loss / len(val_loader)
            val_accuracy = correct_predictions.double() / total_samples

            logger.info(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')
            logger.info(f'Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        fold_accuracies.append(val_accuracy.item())

    # Return the average validation accuracy across folds
    mean_accuracy = np.mean(fold_accuracies)
    logger.info(f'Mean Validation Accuracy: {mean_accuracy:.4f}')
    return -mean_accuracy  # Minimize negative accuracy

# Create an Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)

# Best hyperparameters
logger.info('Best Hyperparameters:')
logger.info(study.best_params)

# Retrain the model with best hyperparameters on the full training set
best_params = study.best_params
weight_decay = best_params['weight_decay']
dropout_rate = best_params['dropout_rate']
learning_rate = best_params['learning_rate']
batch_size = best_params['batch_size']

# Prepare final train and validation sets
from sklearn.model_selection import train_test_split

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels)

train_set = set(train_texts)
val_set = set(val_texts)
overlap = train_set.intersection(val_set)
logger.info(f'Number of overlapping samples: {len(overlap)}')


train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length=512)
val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_length=512)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Initialize model with best hyperparameters
config = DistilBertConfig.from_pretrained(
    'distilbert-base-cased',
    num_labels=num_labels,
    dropout=dropout_rate,
    attention_dropout=dropout_rate
)
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-cased',
    config=config
)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

epochs = 50  # You can adjust the number of epochs
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# Training loop with early stopping
best_val_loss = float('inf')
patience = 4
counter = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for batch in tqdm(train_loader, desc=f'Training Epoch {epoch + 1}'):
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

    avg_train_loss = total_loss / len(train_loader)
    train_accuracy = correct_predictions.double() / total_samples

    # Validation
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f'Evaluating Epoch {epoch + 1}'):
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

    avg_val_loss = total_loss / len(val_loader)
    val_accuracy = correct_predictions.double() / total_samples

    logger.info(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')
    logger.info(f'Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        # Save the best model
        output_dir = './best_model'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f'Best model saved to {output_dir}')
    else:
        counter += 1
        if counter >= patience:
            logger.info('Early stopping triggered.')
            break

# Generate Confusion Matrix and Classification Report
def get_predictions(model, data_loader):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu()
            predictions.extend(preds.numpy())
            true_labels.extend(labels.numpy())
    return predictions, true_labels

val_preds, val_true = get_predictions(model, val_loader)

# Classification report
print('Classification Report:')
print(classification_report(val_true, val_preds, target_names=label_set))

# Confusion matrix
cm = confusion_matrix(val_true, val_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_set)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# t-SNE visualization
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
                output_hidden_states=True
            )

            # Get the last hidden state
            hidden_states = outputs.hidden_states  # Tuple of length num_layers + 1
            last_hidden_state = hidden_states[-1]  # Shape: [batch_size, seq_length, hidden_size]

            # Mean pooling
            pooled_output = torch.mean(last_hidden_state, dim=1)  # Shape: [batch_size, hidden_size]
            features.append(pooled_output.cpu().numpy())
            labels_list.extend(labels_batch.numpy())

    features = np.concatenate(features, axis=0)
    labels_array = np.array(labels_list)
    return features, labels_array

features, labels_array = get_model_outputs(model, val_loader, device)

# Apply t-SNE
logger.info('Applying t-SNE...')
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features)

# Plot t-SNE visualization
plt.figure(figsize=(12, 8))
for label_id in np.unique(labels_array):
    indices = labels_array == label_id
    plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label=id_to_label[label_id], alpha=0.7)
plt.legend()
plt.title('t-SNE of Model Outputs')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True)
plt.show()
