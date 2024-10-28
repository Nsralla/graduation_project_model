import json
import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import os
import sys
import pickle
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

def colored_print(message, color):
    """
    Prints a colored message to the console.
    """
    color_dict = {
        'green': Fore.GREEN,
        'yellow': Fore.YELLOW,
        'red': Fore.RED,
        'blue': Fore.BLUE,
        'cyan': Fore.CYAN
    }
    print(color_dict.get(color, Fore.WHITE) + message + Style.RESET_ALL)

def load_data(jsonl_path):
    """
    Loads text and labels from a JSON Lines (JSONL) file.

    Args:
        jsonl_path (str): Path to the JSONL file.

    Returns:
        texts (list): List of text strings.
        labels (list): List of labels.
    """
    colored_print(f"Loading data from {jsonl_path}...", 'cyan')
    texts = []
    labels = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                try:
                    data = json.loads(line)
                    texts.append(data['text'])
                    labels.append(data['label'])
                except json.JSONDecodeError as jde:
                    colored_print(f"JSON decode error in line {line_number}: {jde}", 'red')
                except KeyError as ke:
                    colored_print(f"Missing key {ke} in line {line_number}", 'red')
        colored_print(f"Successfully loaded {len(texts)} entries.", 'green')
        return texts, labels
    except FileNotFoundError:
        colored_print(f"File {jsonl_path} not found.", 'red')
        sys.exit(1)
    except Exception as e:
        colored_print(f"Error loading data: {e}", 'red')
        sys.exit(1)

def extract_features(texts, tokenizer, model, device, batch_size=32):
    """
    Extracts BERT features for a list of texts.

    Args:
        texts (list): List of text strings.
        tokenizer (BertTokenizer): BERT tokenizer.
        model (BertModel): BERT model.
        device (torch.device): Device to run the model on.
        batch_size (int): Number of samples per batch.

    Returns:
        cls_features (list): List of [CLS] token embeddings (each tensor is [hidden_size]).
        all_token_features (list): List of tensors containing all token embeddings for each batch.
    """
    colored_print("Starting feature extraction...", 'cyan')
    model.to(device)
    model.eval()
    cls_features = []
    all_token_features = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches"):
            batch_texts = texts[i:i+batch_size]
            # Tokenize the batch
            encoded = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Extract the [CLS] token representation
            batch_cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
            # Extract all token embeddings
            batch_token_embeddings = outputs.last_hidden_state.cpu()

            # Append each [CLS] embedding individually
            for cls_embedding in batch_cls_embeddings:
                cls_features.append(cls_embedding)

            # Append the entire batch of token embeddings
            all_token_features.append(batch_token_embeddings)

            colored_print(f"Processed batch {i//batch_size + 1}", 'blue')

    colored_print("Feature extraction completed.", 'green')
    return cls_features, all_token_features

def save_features(cls_features, all_token_features, labels, 
                  cls_features_path, all_token_features_path, labels_path):
    """
    Saves the extracted [CLS] features, all token features, and labels to files.

    Args:
        cls_features (list): List of [CLS] token embeddings.
        all_token_features (list): List of tensors containing all token embeddings.
        labels (list): List of labels.
        cls_features_path (str): Path to save [CLS] features.
        all_token_features_path (str): Path to save all token features.
        labels_path (str): Path to save labels.
    """
    try:
        # Save [CLS] features as a list of tensors
        torch.save(cls_features, cls_features_path)
        colored_print(f"[CLS] features saved to {cls_features_path}", 'green')
    except Exception as e:
        colored_print(f"Error saving [CLS] features: {e}", 'red')

    try:
        # Save all_token_features as a list of tensors
        torch.save(all_token_features, all_token_features_path)
        colored_print(f"All token features saved to {all_token_features_path}", 'green')
    except Exception as e:
        colored_print(f"Error saving all token features: {e}", 'red')

    try:
        with open(labels_path, 'wb') as f:
            pickle.dump(labels, f)
        colored_print(f"Labels saved to {labels_path}", 'green')
    except Exception as e:
        colored_print(f"Error saving labels: {e}", 'red')

def main():
    # Configuration
    jsonl_file = 'transcriptions.jsonl'  # Path to your JSONL file
    output_dir = './bert-base-cased_freezing_firstLayers_20_Best'
    cls_features_output_path = 'cls_features.pt'
    all_token_features_output_path = 'all_token_features.pt'
    labels_output_path = 'labels.pkl'
    batch_size = 4  # Adjust based on your GPU memory

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        colored_print("GPU is available. Using GPU for computations.", 'green')
    else:
        colored_print("GPU not available. Using CPU for computations.", 'yellow')

    # Load tokenizer and model
    colored_print("Loading tokenizer and model...", 'cyan')
    try:
        tokenizer = BertTokenizer.from_pretrained(output_dir)
        model = BertModel.from_pretrained(output_dir)
        colored_print("Tokenizer and model loaded successfully.", 'green')
    except Exception as e:
        colored_print(f"Error loading tokenizer or model: {e}", 'red')
        sys.exit(1)

    print(model.config)

    # Load data
    texts, labels = load_data(jsonl_file)
    print(f"label length: {len(labels)}")

    # Validate labels
    valid_labels = {'A1', 'A2', 'B1_1', 'B1_2', 'B2', 'C1', 'C2'}
    if not all(label in valid_labels for label in labels):
        invalid_labels = set(labels) - valid_labels
        colored_print(f"Error: Some labels are invalid: {invalid_labels}", 'red')
        sys.exit(1)
    else:
        colored_print("All labels are valid.", 'green')

    # Extract features
    cls_features, all_token_features = extract_features(texts, tokenizer, model, device, batch_size=batch_size)

    # Save features and labels
    save_features(cls_features, all_token_features, labels, 
                  cls_features_output_path, all_token_features_output_path, labels_output_path)

    colored_print("All tasks completed successfully!", 'green')

if __name__ == "__main__":
    main()
