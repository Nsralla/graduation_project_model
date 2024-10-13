import torch
import gc
import os
import librosa
from padding import logger
import torch
import math

def process_text(audio_file_path, tokenizer, bert_model, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1  # Reduce the batch size to minimize memory usage

    # Check if the specified audio file exists
    if not os.path.isfile(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    # Load the audio file using librosa
    audio_data, sample_rate = librosa.load(audio_file_path, sr=16000, mono=True)
    total_duration = librosa.get_duration(y=audio_data, sr=sample_rate)

    # Define the segment length (30 seconds)
    segment_length = 30  # seconds

    # If the audio duration is longer than the segment length, split it
    segments = []
    if total_duration > segment_length:
        num_segments = math.ceil(total_duration / segment_length)
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
        # Retry mechanism to handle OOM errors
        success = False
        retries = 0
        max_retries = 3  # You can adjust this based on your needs
        
        while not success and retries < max_retries:
            try:
                # Transcribe the audio segment
                result = model.transcribe(segment, batch_size=batch_size, language="en")

                # Extract the transcribed text from the segments
                segment_text = " ".join([seg['text'] for seg in result["segments"]])
                extracted_text += " " + segment_text
                logger.info(f"Extracted text: {segment_text}")
                logger.info("--------------------------------------");

                # Free up GPU memory after each segment processing
                torch.cuda.empty_cache()
                gc.collect()

                success = True  # Mark successful processing

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.warning(f"Out of memory error during transcription. Retrying... ({retries+1}/{max_retries})")
                    retries += 1
                    torch.cuda.empty_cache()
                    gc.collect()

                    # If retries are exhausted, fallback to CPU
                    if retries == max_retries:
                        logger.error("Max retries exceeded. Falling back to CPU processing.")
                        device = torch.device("cpu")
                        model.to(device)
                else:
                    # For other errors, raise them
                    logger.error(f"RuntimeError during transcription: {e}")
                    raise

    # Tokenize the extracted text and move to the correct device
    inputs = tokenizer(extracted_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Extract features from the text using BERT
    with torch.no_grad():
        outputs = bert_model(**inputs)
        features = outputs.last_hidden_state.cpu()  # Move features to CPU to free up GPU memory

    # Clean up resources to free up memory
    del inputs, outputs
    torch.cuda.empty_cache()
    gc.collect()

    return features