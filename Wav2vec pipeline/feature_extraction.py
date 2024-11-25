import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Processor,
)
from tqdm import tqdm
from dataclasses import dataclass
from typing import Union
import gc
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Set up random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Function to extract label from the filename
def extract_label_from_filename(filename):
    filename = filename.lower()
    possible_labels = ['a1', 'c1', 'c2']
    for label in possible_labels:
        if label in filename:
            return label
    print(f"No label found in filename: {filename}")
    return None

# Define the Dataset class
class AudioDataset(Dataset):
    def __init__(self, audio_entries, processor):
        self.audio_entries = audio_entries
        self.processor = processor

    def __len__(self):
        return len(self.audio_entries)

    def __getitem__(self, idx):
        entry = self.audio_entries[idx]
        audio_path = entry['audio_path']
        filename = entry['filename']
        label_id = entry['label_id']

        # Load audio
        audio_input, sample_rate = torchaudio.load(audio_path)

        # Resample if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio_input = resampler(audio_input)
            sample_rate = 16000

        # Convert to mono if necessary
        if audio_input.shape[0] > 1:
            audio_input = torch.mean(audio_input, dim=0, keepdim=True)

        audio_input = audio_input.squeeze()

        # Get the input values from the processor
        input_values = self.processor(audio_input.numpy(), sampling_rate=16000).input_values[0]

        return {
            'input_values': torch.tensor(input_values, dtype=torch.float),
            'filename': filename,
            'label': torch.tensor(label_id, dtype=torch.long)
        }

# Define data collator
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features):
        input_values = [f['input_values'] for f in features]
        labels = torch.tensor([f['label'] for f in features], dtype=torch.long)
        filenames = [f['filename'] for f in features]

        batch = self.processor.pad(
            {"input_values": input_values},
            padding=self.padding,
            return_tensors="pt"
        )
        batch['labels'] = labels
        batch['filenames'] = filenames
        return batch

if __name__ == '__main__':
    # Update the path to the audio files
    audio_base_dir = r'Youtube\testing youtube'

    # Directory to save extracted features
    feature_save_dir = r'Youtube\extracted features\testing features'
    os.makedirs(feature_save_dir, exist_ok=True)
    
    output_dir = "Youtube\wav2vec2_finetuned_checkpoints\epoch_4"

    # Load the processor from your saved checkpoint
    processor = Wav2Vec2Processor.from_pretrained(output_dir)

    # Load the model from your saved checkpoint
    model = Wav2Vec2ForSequenceClassification.from_pretrained(output_dir)

    # Set the device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print("Model and processor loaded successfully.")

    # Build audio entries with file paths and labels
    audio_entries = []
    for root, _, files in os.walk(audio_base_dir):
        for file in files:
            if file.endswith('.mp3') or file.endswith('.wav'):
                label = extract_label_from_filename(file)
                if label is not None:
                    audio_entries.append({
                        'audio_path': os.path.join(root, file),
                        'filename': file,
                        'label': label
                    })
                else:
                    print(f"Skipping file {file} as no label was found.")

    print(f"Total audio files found: {len(audio_entries)}")

    # Create label mappings
    label_set = sorted(set(entry['label'] for entry in audio_entries))
    label_to_id = {label: idx for idx, label in enumerate(label_set)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    print(f"Label to ID mapping: {label_to_id}")

    # Update entries with label IDs
    for entry in audio_entries:
        entry['label_id'] = label_to_id[entry['label']]

    # Create DataLoader
    batch_size = 1  # Adjust based on your GPU memory
    dataset = AudioDataset(audio_entries, processor)
    data_collator = DataCollatorCTCWithPadding(processor=processor)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle for feature extraction
        num_workers=1,  # Adjust based on your CPU cores
        collate_fn=data_collator,
        pin_memory=True
    )

    # Feature extraction loop
    print("Starting feature extraction...")

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Extracting and Saving Features'):
            try:
                input_values = batch['input_values'].to(device, non_blocking=True)
                filenames = batch['filenames']
                labels = batch['labels']

                # Forward pass with output_hidden_states=True
                outputs = model(input_values=input_values, output_hidden_states=True)

                # Extract the last hidden state (before the classification head)
                hidden_states = outputs.hidden_states[-1]  # Shape: [batch_size, seq_len, hidden_size]

                # Iterate over the batch and save features individually
                for i in range(len(filenames)):
                    audio_feature = hidden_states[i].cpu().numpy()  # Shape: [seq_len, hidden_size]
                    filename = filenames[i]
                    filename_no_ext = os.path.splitext(filename)[0]
                    save_filename = f"{filename_no_ext}.npz"
                    save_path = os.path.join(feature_save_dir, save_filename)

                    # Save the feature to a .npz file
                    np.savez_compressed(
                        save_path,
                        feature=audio_feature,  # Shape: [seq_len, hidden_size]
                        filename=filename_no_ext,
                        label=labels[i].item()
                    )

                # Clean up
                del input_values, outputs, hidden_states
                torch.cuda.empty_cache()
                gc.collect()

            except torch.cuda.OutOfMemoryError:
                print("Out of memory on GPU. Trying to process batch on CPU.")
                torch.cuda.empty_cache()
                gc.collect()
                # Move model and data to CPU
                model_cpu = model.to('cpu')
                input_values_cpu = batch['input_values'].to('cpu', non_blocking=True)
                labels = batch['labels']
                # Process on CPU
                outputs = model_cpu(input_values=input_values_cpu, output_hidden_states=True)

                hidden_states = outputs.hidden_states[-1]  # Shape: [batch_size, seq_len, hidden_size]

                for i in range(len(filenames)):
                    audio_feature = hidden_states[i].numpy()  # On CPU
                    filename = filenames[i]
                    filename_no_ext = os.path.splitext(filename)[0]
                    save_filename = f"{filename_no_ext}.npz"
                    save_path = os.path.join(feature_save_dir, save_filename)

                    # Save the feature to a .npz file
                    np.savez_compressed(
                        save_path,
                        feature=audio_feature,  # Shape: [seq_len, hidden_size]
                        filename=filename_no_ext,
                        label=labels[i].item()
                    )

                # Clean up
                del input_values_cpu, outputs, hidden_states
                gc.collect()
                # Move model back to device
                model = model_cpu.to(device)
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"An error occurred: {e}")
                # Handle other exceptions if needed

    print("Feature extraction and saving complete.")

    # Load extracted features and apply t-SNE
    print("Loading extracted features for t-SNE visualization...")

    # Collect all feature files
    feature_files = [os.path.join(feature_save_dir, f) for f in os.listdir(feature_save_dir) if f.endswith('.npz')]

    features_list = []
    labels_list = []

    for feature_file in tqdm(feature_files, desc='Loading Features'):
        data = np.load(feature_file)
        feature = data['feature']  # Shape: [seq_len, hidden_size]
        filename = data['filename']
        label_id = data['label']
        # Optionally, we can pool the features over time, e.g., take the mean over time
        feature_pooled = feature.mean(axis=0)  # Shape: [hidden_size]
        features_list.append(feature_pooled)
        labels_list.append(label_id)

    features_array = np.stack(features_list)
    labels_array = np.array(labels_list)

    # Apply t-SNE
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features_array)

    # Plotting
    plt.figure(figsize=(12, 8))
    for label_id in np.unique(labels_array):
        indices = labels_array == label_id
        plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label=id_to_label[label_id], alpha=0.7)
    plt.legend()
    plt.title('t-SNE of Extracted Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.show()
