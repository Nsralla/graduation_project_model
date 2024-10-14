import torch
import traceback
import os
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from append_IELTS_set import get_IELTS_audio_files
from audioProcessing import process_single_audio
from IELTSDatasetClass import extract_label_from_path
from padding import logger

# Set up logging
log_file = 'processing_log.txt'
# Collect all audio file paths and extract their labels
IELTS_FILES = get_IELTS_audio_files()
class_labels = ['A1', 'A2', 'B1_1', 'B1_2', 'B2', 'C1', 'C2']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths for saving intermediate results
output_folder = 'intermediate_results'
os.makedirs(output_folder, exist_ok=True)
intermediate_file = os.path.join(output_folder, 'features_labels.pt')
processed_files_log = os.path.join(output_folder, 'processed_files.txt')

# Load processed files log
if os.path.exists(processed_files_log):
    with open(processed_files_log, 'r') as f:
        processed_files = set(f.read().splitlines())
else:
    processed_files = set()

def extract_features_labels(save_every=40):
    features_list = []
    labels = []

    # Load Wav2Vec 2.0 model and processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-large-960h",
        output_hidden_states=True,
        output_attentions=True
    ).to(device)
    model.eval()

    for i, audio_file in enumerate(IELTS_FILES):
        if audio_file in processed_files:
            logger.info(f"Skipping already processed file: {audio_file}")
            continue

        logger.info(f"Processing file {i+1}/{len(IELTS_FILES)}: {audio_file}")
        try:
            label = extract_label_from_path(audio_file)
            label_index = class_labels.index(label)
            logger.info(f"Extracted label: {label}, Label index: {label_index}")

            # Step 3: Process Audio and extract features
            audio_features = process_single_audio(audio_file, processor, model)
            audio_features = audio_features.cpu()  # Move to CPU

            features_list.append(audio_features)
            labels.append(label_index)

            # Log processed file
            with open(processed_files_log, 'a') as f:
                f.write(audio_file + '\n')

            # Save intermediate results every N files
            if (i + 1) % save_every == 0:
                save_intermediate_results(features_list, labels, intermediate_file)
                logger.info(f"Intermediate results saved at iteration {i+1}.")
                # Clear lists after saving
                features_list = []
                labels = []

            logger.debug(f"Current size of features_list: {len(features_list)}")
            logger.debug(f"Current size of labels list: {len(labels)}")
            logger.info("-------------------------------------------------------")

            # Clear GPU memory after processing
            del audio_features
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error processing file {audio_file}: {e}")
            logger.debug(traceback.format_exc())
        
        # Additional GPU memory management
        torch.cuda.empty_cache()
    
    # Save remaining results after the loop finishes
    if features_list and labels:
        save_intermediate_results(features_list, labels, intermediate_file)

    return features_list, labels

def save_intermediate_results(features_list, labels, file_path):
    # If file already exists, load it and append new data
    if os.path.exists(file_path):
        try:
            existing_data = torch.load(file_path)
            existing_features = existing_data.get('features', [])
            existing_labels = existing_data.get('labels', [])
            updated_features = existing_features + features_list
            updated_labels = existing_labels + labels
        except Exception as e:
            logger.error(f"Error loading intermediate file {file_path}: {e}")
            updated_features = features_list
            updated_labels = labels
    else:
        updated_features = features_list
        updated_labels = labels

    data_to_save = {'features': updated_features, 'labels': updated_labels}
    torch.save(data_to_save, file_path)
    logger.info(f"Intermediate results saved to {file_path}.")
