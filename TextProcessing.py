import torch
import gc
import os
import librosa
from padding import logger

def process_text(audio_file_path, tokenizer, bert_model, model):
    # Set the device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1  # Reduce the batch size to minimize memory usage

    # Check if the specified audio file exists
    if not os.path.isfile(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    # Load the audio file using librosa
    audio_data, sample_rate = librosa.load(audio_file_path, sr=16000)  # Convert audio to numpy array
    total_duration = librosa.get_duration(y=audio_data, sr=sample_rate)

    # Define the segment length (1 minute and 30 seconds)
    segment_length = 30  # seconds

    # If the audio duration is longer than the segment length, split it
    segments = []
    if total_duration > segment_length:
        num_segments = int(total_duration // segment_length) + 1
        for i in range(num_segments):
            start = i * segment_length
            end = min((i + 1) * segment_length, total_duration)
            segment = audio_data[int(start * sample_rate):int(end * sample_rate)]
            segments.append(segment)
    else:
        segments.append(audio_data)

    # Initialize an empty string to store the extracted text
    extracted_text = ""

    # Process each segment
    for segment in segments:
        # Transcribe the audio segment
        try:
            result = model.transcribe(segment, batch_size=batch_size)
        except RuntimeError as e:
            logger.error(f"RuntimeError during transcription: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            continue

        # Extract the transcribed text from the segments
        segment_text = " ".join([seg['text'] for seg in result["segments"]])
        extracted_text += " " + segment_text
        logger.info(f"Extracted text: {segment_text}")
        logger.info("--------------------------------------");

        # Free up GPU memory after each segment processing
        torch.cuda.empty_cache()
        gc.collect()

    # Tokenize the extracted text and move to the correct device
    inputs = tokenizer(extracted_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Extract features from the text using BERT
    with torch.no_grad():
        outputs = bert_model(**inputs)
        features = outputs.last_hidden_state.cpu()  # Move features to CPU to free up GPU memory
        
    # Extract [CLS] token (usually the first token)
    cls_token = features[:, 0, :]  # Shape: [batch_size, feature_dim]
        
    # Clean up resources to free up memory
    del inputs, outputs
    torch.cuda.empty_cache()
    gc.collect()

    return features, cls_token