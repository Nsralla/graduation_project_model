import torch
import pickle
import traceback
import logging
import torch.nn as nn
from transformers import XLNetTokenizer, XLNetModel, Wav2Vec2Processor, Wav2Vec2Model
from append_IELTS_set import get_IELTS_audio_files
from TextProcessing import process_text
from audioProcessing import process_single_audio
from IELTSDatasetClass import extract_label_from_path
import whisperx
from padding import logger

# Collect all audio file paths and extract their labels
IELTS_FILES = get_IELTS_audio_files()
class_labels = ['A1', 'A2', 'B1_1', 'B1_2', 'B2', 'C1', 'C2']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
url = 'cls_tokens_extracted'

def extract_features_labels():
    features_list = []
    all_concatenated_cls_tokens = []
    labels = []

    # Load Wav2Vec 2.0 model and processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-large-960h",
        output_hidden_states=True,
        output_attentions=True
    ).to(device)
    model.eval()

    # Load pre-trained XLNet model and tokenizer
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
    bert_model = XLNetModel.from_pretrained('xlnet-large-cased').to(device)

    compute_type = "float16"  # Use float16 to reduce memory
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    model_bert = whisperx.load_model("small", device=device_str, compute_type=compute_type)

    for i, audio_file in enumerate(IELTS_FILES[501:1000]):
        logger.info(f"Processing file {i+1}/{len(IELTS_FILES)}: {audio_file}")
        try:
            label = extract_label_from_path(audio_file)
            label_index = class_labels.index(label)
            logger.info(f"Extracted label: {label}, Label index: {label_index}")

            # Step 2: Process Text
            text_features, text_cls_token = process_text(audio_file, tokenizer, bert_model, model_bert)
            text_features = text_features.to(device)
            text_cls_token = text_cls_token.cpu().detach()  # Move to CPU and detach
            logger.info(f"Text features extracted. Shape: {text_features.shape}")
            logger.info(f"Text CLS token shape: {text_cls_token.shape}")

            # Step 3: Process Audio
            audio_features, audio_cls_token = process_single_audio(audio_file, processor, model)
            audio_features = audio_features.to(device)
            audio_cls_token = audio_cls_token.cpu().detach()  # Move to CPU and detach

            if audio_features.dim() == 2:
                audio_features = audio_features.unsqueeze(0)

            logger.info(f"Audio features extracted. Shape: {audio_features.shape}")
            logger.info(f"Audio CLS token shape: {audio_cls_token.shape}")

            # Step 3.1: Concatenate the CLS tokens on the CPU
            concatenated_cls = torch.cat((text_cls_token, audio_cls_token), dim=1)
            all_concatenated_cls_tokens.append({
                'concatenated_cls': concatenated_cls,
                'label': label
            })

            logger.info(f"Concatenated CLS tokens shape: {concatenated_cls.shape}")

            # Step 4: Downsample Audio Features
            pooling_layer = nn.AdaptiveAvgPool1d(text_features.shape[1])
            audio_features_downsampled = pooling_layer(audio_features.permute(0, 2, 1)).permute(0, 2, 1).detach().cpu()
            logger.info(f"Audio features after pooling. Shape: {audio_features_downsampled.shape}")

            # Step 5: Concatenate Text and Audio Features
            concatenated_features = torch.cat((text_features.cpu(), audio_features_downsampled), dim=1).detach().cpu()
            logger.info(f"Concatenated features shape: {concatenated_features.shape}")

            features_list.append(concatenated_features)
            labels.append(label_index)

            logger.debug(f"Current size of features_list: {len(features_list)}")
            logger.debug(f"Current size of labels list: {len(labels)}")
            logger.info("-------------------------------------------------------")

            # Clear GPU memory after processing
            del text_features, audio_features, audio_features_downsampled
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error processing file {audio_file}: {e}")
            logger.debug(traceback.format_exc())
        
        # Additional GPU memory management
        torch.cuda.empty_cache()

    # After the loop finishes, save the accumulated concatenated CLS tokens to a file
    torch.save(all_concatenated_cls_tokens, f'{url}\\all_cls_tokens_concatenated2.pt')
    return features_list, labels