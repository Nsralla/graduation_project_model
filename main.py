import torch
from padding import logger
from extract_features import extract_features_labels
import os
import time

torch.cuda.empty_cache()
# Define paths
folder_path = 'process_text_audio_seperatly\\text_features_only'
file_path = os.path.join(folder_path, 'text_features_and_labels_second_try.pt')

# Helper function to create a fallback file path
def generate_fallback_file_path(folder_path):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(folder_path, f'text_features_and_labels_backup_{timestamp}.pt')

# STEP 1: EXTRACT FEATURES AND LABELS FROM AUDIOS.
features_list, labels = extract_features_labels()

# Check if features_list and labels have the same length
if len(features_list) != len(labels):
    logger.error(f"Inconsistent dataset sizes: features_list={len(features_list)}, labels={len(labels)}")
    raise ValueError("Mismatch between features and labels.")

# Ensure the folder exists before saving
if not os.path.isdir(folder_path):
    try:
        os.makedirs(folder_path, exist_ok=True)
        logger.info(f"Created folder: {folder_path}")
    except Exception as e:
        logger.error(f"Failed to create folder '{folder_path}': {e}")
        raise

# Try loading the existing data, with fallback in case of an error
try:
    if os.path.exists(file_path):
        # Attempt to load existing data
        existing_data = torch.load(file_path)
        existing_features = existing_data.get('features', [])
        existing_labels = existing_data.get('labels', [])

        # Append new features and labels to existing data
        updated_features = existing_features + features_list
        updated_labels = existing_labels + labels

        logger.info(f"Loaded existing data from '{file_path}', appending new features and labels.")
    else:
        # If no existing file is found, start fresh
        logger.info(f"No existing file found. Creating new dataset.")
        updated_features = features_list
        updated_labels = labels

    # Save updated features and labels to the original file
    data_to_save = {'features': updated_features, 'labels': updated_labels}
    torch.save(data_to_save, file_path)
    logger.info(f"Updated features and labels saved successfully to '{file_path}'.")

except Exception as e:
    # Handle any errors during loading or appending
    logger.error(f"Failed to load or append data to '{file_path}': {e}")
    
    # Generate a fallback file name and save new features there
    fallback_file_path = generate_fallback_file_path(folder_path)
    
    try:
        # Save only the new features and labels to the fallback file
        data_to_save = {'features': features_list, 'labels': labels}
        torch.save(data_to_save, fallback_file_path)
        logger.info(f"Due to an error, new features and labels were saved to '{fallback_file_path}' instead.")
    except Exception as save_error:
        logger.error(f"Failed to save new features and labels to fallback file '{fallback_file_path}': {save_error}")
        raise
