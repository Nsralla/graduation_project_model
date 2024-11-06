import json
import torch
from transformers import BertTokenizer,  BertModel, BertForSequenceClassification
from tqdm import tqdm
import os
import sys
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
    Loads texts, labels, and filenames from a JSON Lines (JSONL) file.

    Args:
        jsonl_path (str): Path to the JSONL file.

    Returns:
        texts (list): List of text strings.
        labels (list): List of labels.
        filenames (list): List of filenames.
    """
    colored_print(f"Loading data from {jsonl_path}...", 'cyan')
    texts = []
    labels = []
    filenames = []
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
                    filenames.append(data['filename'])
                except json.JSONDecodeError as jde:
                    colored_print(f"JSON decode error in line {line_number}: {jde}", 'red')
                except KeyError as ke:
                    colored_print(f"Missing key {ke} in line {line_number}", 'red')
        colored_print(f"Successfully loaded {len(texts)} entries.", 'green')
        return texts, labels, filenames
    except FileNotFoundError:
        colored_print(f"File {jsonl_path} not found.", 'red')
        sys.exit(1)
    except Exception as e:
        colored_print(f"Error loading data: {e}", 'red')
        sys.exit(1)

def extract_and_save_features(texts, filenames, tokenizer, model, device, output_dir):
    """
    Extracts BERT features for each text and saves them individually.

    Args:
        texts (list): List of text strings.
        filenames (list): List of filenames corresponding to each text.
        tokenizer (BertTokenizer): BERT tokenizer.
        model (BertModel): BERT model.
        device (torch.device): Device to run the model on.
        output_dir (str): Directory to save the features.
    """
    colored_print("Starting feature extraction and saving...", 'cyan')
    model.to(device)
    model.eval()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with torch.no_grad():
        for text, filename in tqdm(zip(texts, filenames), desc="Processing Texts", total=len(texts)):
            # Tokenize the text
            encoded = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Extract all token embeddings
            token_embeddings = outputs.last_hidden_state.cpu()  # Shape: [1, seq_len, hidden_size]

            # Save the token embeddings to a file named after the original filename
            feature_filename = os.path.splitext(filename)[0] + '.pt'  # Replace extension with .pt
            feature_path = os.path.join(output_dir, feature_filename)

            # Save the tensor to the feature_path
            torch.save(token_embeddings.squeeze(0), feature_path)  # Remove batch dimension

            colored_print(f"Features saved for {filename} -> {feature_filename}", 'blue')

    colored_print("Feature extraction and saving completed.", 'green')

def main():
    # Configuration
    jsonl_file = 'transcriptions.jsonl'  # Path to your JSONL file
    output_features_dir = './extracted_features_from_ICNALE_base_model'  # Directory to save individual feature files
    batch_size = 1  

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        colored_print("GPU is available. Using GPU for computations.", 'green')
    else:
        colored_print("GPU not available. Using CPU for computations.", 'yellow')

    # Load tokenizer and model
    colored_print("Loading tokenizer and model...", 'cyan')
    output_dir = './bert_finetuned_epoch_9'  # Directory where the model and tokenizer were saved

    try:
        tokenizer = BertTokenizer.from_pretrained(output_dir)
        model = BertModel.from_pretrained(output_dir)  # Use BertModel to access token embeddings
        print("Tokenizer and model loaded successfully.")
    except Exception as e:
        print(f"Error loading tokenizer or model: {e}")
        sys.exit(1)


    # Load data
    texts, labels, filenames = load_data(jsonl_file)
    print(f"Number of labels: {len(labels)}")

    # Validate labels (optional)
    valid_labels = {'A2', 'B1_1', 'B1_2', 'B2'}  # Update based on your actual labels
    if not all(label in valid_labels for label in labels):
        invalid_labels = set(labels) - valid_labels
        colored_print(f"Error: Some labels are invalid: {invalid_labels}", 'red')
        sys.exit(1)
    else:
        colored_print("All labels are valid.", 'green')

    # Extract features and save them individually
    extract_and_save_features(texts, filenames, tokenizer, model, device, output_features_dir)

    colored_print("All tasks completed successfully!", 'green')

if __name__ == "__main__":
    main()
