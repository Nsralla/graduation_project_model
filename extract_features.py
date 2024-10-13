import torch
import traceback
from transformers import XLNetTokenizer, XLNetModel
from append_IELTS_set import get_IELTS_audio_files
from TextProcessing import process_text
from IELTSDatasetClass import extract_label_from_path
from padding import logger
import whisperx

# Collect all audio file paths and extract their labels
IELTS_FILES = get_IELTS_audio_files()  # Ensure this is defined
class_labels = ['A1', 'A2', 'B1_1', 'B1_2', 'B2', 'C1', 'C2']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

def extract_features_labels():
    features_list = []
    labels = []
    
    # Load pre-trained XLNet model and tokenizer
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
    bert_model = XLNetModel.from_pretrained('xlnet-large-cased').to(device)
    
    # Load Whisper model on GPU or CPU
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model = whisperx.load_model("medium", device=device_str)  # Whisper model loaded for GPU if available

    # Disable gradient tracking to save memory
    with torch.no_grad():
        for i, audio_file in enumerate(IELTS_FILES[1501:2500]):
            logger.info(f"Processing file {i+1}/{len(IELTS_FILES)}: {audio_file}")
            try:
                # Extract label
                label = extract_label_from_path(audio_file)
                label_index = class_labels.index(label)
                logger.info(f"Extracted label: {label}, Label index: {label_index}")

                # Step 2: Process Text
                # Ensure audio is processed with Whisper and XLNet on the same device
                text_features = process_text(audio_file, tokenizer, bert_model, whisper_model)
                
                # Move the extracted features to the same device (if not already)
                text_features = text_features.to(device)
                logger.info(f"Text features extracted. Shape: {text_features.shape}")

                # Move features to CPU and append to list to free up GPU memory
                features_list.append(text_features.cpu())
                labels.append(label_index)

                logger.debug(f"Current size of features_list: {len(features_list)}")
                logger.debug(f"Current size of labels list: {len(labels)}")
                logger.info("-------------------------------------------------------")

                # Clear GPU memory after processing each file
                del text_features
                torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error processing file {audio_file}: {e}")
                logger.debug(traceback.format_exc())

            # Additional GPU memory management
            torch.cuda.empty_cache()

    return features_list, labels
