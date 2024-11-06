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
    possible_labels = ['a2', 'b1_1', 'b1_2', 'b2']
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
    audio_base_dir = r'D:\Graduation_Project\testing_icnale'

    # Path to your saved model checkpoint directory
    checkpoint_dir = './Second_try_more_freezing_layers/epoch_9'  # Update this path if necessary

    # Directory to save extracted features
    feature_save_dir = './Second_try_more_freezing_layers/ICNALE_features_testing_dataset'
    os.makedirs(feature_save_dir, exist_ok=True)

    # Load the processor from your saved checkpoint
    processor = Wav2Vec2Processor.from_pretrained(checkpoint_dir)

    # Load the model from your saved checkpoint
    model = Wav2Vec2ForSequenceClassification.from_pretrained(checkpoint_dir)

    # Set the device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print("Model and processor loaded successfully from custom checkpoint.")

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
        pin_memory=True,
        persistent_workers=True
    )

    # Feature extraction loop
    print("Starting feature extraction...")

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Extracting and Saving Features'):
            input_values = batch['input_values'].to(device, non_blocking=True)
            filenames = batch['filenames']

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
                    filename=filename_no_ext
                )

    print("Feature extraction and saving complete.")
