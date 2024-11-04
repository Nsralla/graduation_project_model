# Install necessary packages (if not already installed)
# !pip install transformers torchaudio

# Import necessary libraries
import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from tqdm.notebook import tqdm
import json

# Step 1: Mount Google Drive to access files if necessary
# from google.colab import drive
# drive.mount('/content/drive')

# Step 2: Define paths
# Path to the JSONL file with transcriptions
jsonl_file_path = '/content/drive/MyDrive/transcriptions.jsonl'  # Update this path if necessary

# Path to the audio files
audio_base_dir = '/content/drive/MyDrive/audios/audios'  # Update this path if necessary

# Path to the model checkpoint and processor
checkpoint_dir = '/content/drive/MyDrive/your_project/checkpoints'  # Update this path if necessary
model_checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_21.pt"
processor_dir = f"{checkpoint_dir}/processor_epoch_21"

# Set up directory to save extracted features
feature_save_dir = '/content/extracted_features'
os.makedirs(feature_save_dir, exist_ok=True)

# Step 3: Load the processor
processor = Wav2Vec2Processor.from_pretrained(processor_dir)

# Step 4: Load the model
from transformers import Wav2Vec2Config

# Ensure num_labels matches your use case
num_labels = 7  # Update based on your labels
config = Wav2Vec2Config.from_pretrained('facebook/wav2vec2-base', num_labels=num_labels)

# Initialize the model
model = Wav2Vec2ForSequenceClassification.from_pretrained('facebook/wav2vec2-base', config=config)

# Load the model state dict
checkpoint = torch.load(model_checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model.load_state_dict(checkpoint['model_state_dict'])

# Move the model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()  # Set model to evaluation mode

# Enable output_hidden_states to get hidden states from the model
model.config.output_hidden_states = True

# Step 5: Load the JSONL file
# Read the JSONL file
transcriptions = []
with open(jsonl_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line.strip())
        transcriptions.append(entry)

print(f"Total transcriptions loaded: {len(transcriptions)}")

# Step 6: Prepare audio entries list
# Build a list of audio file paths based on the transcriptions
audio_entries = []
missing_files = []

for entry in transcriptions:
    filename = entry['filename']  # Use 'filename' key
    label = entry['label']        # Use 'label' key
    audio_path = os.path.join(audio_base_dir, filename)
    if os.path.exists(audio_path):
        audio_entries.append({
            'audio_path': audio_path,
            'filename': filename,
            'label': label
        })
    else:
        missing_files.append(filename)

print(f"Total audio files to process: {len(audio_entries)}")
if missing_files:
    print(f"Missing audio files: {missing_files}")

# Step 7: Define the Dataset class
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
        label = entry['label']

        audio_input, sample_rate = torchaudio.load(audio_path)

        # Resample if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio_input = resampler(audio_input)
            sample_rate = 16000

        # Convert to mono if necessary
        if audio_input.shape[0] > 1:
            audio_input = torch.mean(audio_input, dim=0, keepdim=True)

        # Get the input values from the processor
        input_values = self.processor(audio_input.numpy(), sampling_rate=16000).input_values[0]

        return {
            'input_values': torch.tensor(input_values, dtype=torch.float),
            'filename': filename,
            'label': label
        }

# Step 8: Create DataLoader
# Set batch_size=1 to avoid padding and ensure compatibility
batch_size = 1
dataset = AudioDataset(audio_entries, processor)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Step 9: Extract features and save individually
with torch.no_grad():
    for batch in tqdm(data_loader, desc='Extracting and Saving Features'):
        # Extract data from the batch
        input_values = batch['input_values'][0].to(device)  # Shape: [sequence_length]
        filename = batch['filename'][0]
        label = batch['label'][0]

        # Reshape input_values to [1, sequence_length] to add batch dimension
        input_values = input_values.unsqueeze(0)  # Shape: [1, sequence_length]

        # Forward pass with output_hidden_states=True
        outputs = model(input_values=input_values, output_hidden_states=True)

        # Extract the last hidden state (before the classification head)
        hidden_states = outputs.hidden_states
        last_hidden_state = hidden_states[-1]  # Shape: [1, sequence_length, hidden_size]

        # Convert to CPU numpy array
        audio_feature = last_hidden_state.cpu().numpy()  # Shape: [1, sequence_length, hidden_size]

        # Save the feature to a .npz file
        save_filename = f"{os.path.splitext(filename)[0]}.npz"
        save_path = os.path.join(feature_save_dir, save_filename)

        np.savez_compressed(save_path,
                            feature=audio_feature,  # Shape: [1, sequence_length, hidden_size]
                            filename=filename,
                            label=label)

        # Optionally, download the file if in Colab
        # Uncomment the following lines if you wish to download each file immediately
        from google.colab import files
        files.download(save_path)

        # Print a message (optional)
        print(f"Saved features for {filename} to {save_path}")
