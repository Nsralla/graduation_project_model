from padding import logger
from append_IELTS_set import get_IELTS_audio_files
import torch
import whisperx
from TextProcessing import process_text
from audioProcessing import process_single_audio
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from IELTSDatasetClass import extract_label_from_path
from transformers import BertTokenizer, BertModel
# Collect all audio file paths and extract their labels
IELTS_FILES = get_IELTS_audio_files()
features_list = []
labels = []
class_labels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_features_labels():
    
      # Load Wav2Vec 2.0 model and processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-base-960h",
        output_hidden_states=True,   # Enable hidden states output
        output_attentions=True       # Enable attentions output
    ).to(device)
    
    
    compute_type = "float16"  # Use float16 to reduce memory
    # Load pre-trained BERT model and tokenizer from Hugging Face Transformers
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
        # Convert the torch.device to a string for compatibility
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    model_bert = whisperx.load_model("base", device=device_str, compute_type=compute_type)  # Use "base" model for less GPU usage
    
    # Step 1: Extract features and labels for all files
    for i, audio_file in enumerate(IELTS_FILES[1297:1480]):
        logger.info(f"Processing file {i+1}/{len(IELTS_FILES)}: {audio_file}")

        try:
            # Extract the label (score) from the audio file path
            label = extract_label_from_path(audio_file)
            label_index = class_labels.index(label)
            logger.info(f"Extracted label: {label}, Label index: {label_index}")

            # Step 2: Process Text
            text_features = process_text(audio_file, tokenizer, bert_model, model_bert).to(device)
            logger.info(f"Text features extracted. Shape: {text_features.shape}")  # Should be [1, X, 768]

            # # Step 3: Process Audio
            audio_features = process_single_audio(audio_file, processor, model)[0].to(device)
            if len(audio_features.shape) == 2:
                # Add batch dimension if missing
                audio_features = audio_features.unsqueeze(0)  # Shape becomes [1, T, 768]

            logger.info(f"Audio features extracted. Shape: {audio_features.shape}")

            # # Step 4: Downsample Audio Features
            pooling_layer = nn.AdaptiveAvgPool1d(text_features.shape[1])  # Set target size 
            audio_features_downsampled = pooling_layer(audio_features.permute(0, 2, 1)).permute(0, 2, 1)  # Shape becomes [1, T, X]

            logger.info(f"Audio features after pooling. Shape: {audio_features_downsampled.shape}")
            # # Step 5: Concatenate Text and Audio Features
            concatenated_features = torch.cat((text_features, audio_features_downsampled), dim=1)  # Shape: [1, 2X, 768]
            logger.info(f"Concatenated features shape: {concatenated_features.shape}")

            features_list.append(concatenated_features)
            labels.append(label_index)
            logger.info(f"Stored features for file: {audio_file}")
            logger.debug(f"Current size of features_list: {len(features_list)}")
            logger.debug(f"Current size of labels list: {len(labels)}")

        except Exception as e:
            logger.error(f"Error processing file {audio_file}: {e}")
    return features_list, labels
