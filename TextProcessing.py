import whisperx
import torch
import gc
import os
import librosa

# WHISPER + BERT CODE
def process_text(audio_file_path, tokenizer, bert_model, model):
    """
    Main function to transcribe audio using WhisperX model,
    and extract features from the text using BERT.
    """
    # Set the device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4  # Adjust according to your system memory

    # Check if the specified audio file exists
    if not os.path.isfile(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    # Load the audio file using librosa
    audio_data, sample_rate = librosa.load(audio_file_path, sr=16000)
    total_duration = librosa.get_duration(y=audio_data, sr=sample_rate)

    # Define the segment length (1 minute and 30 seconds)
    segment_length = 90  # seconds

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
        result = model.transcribe(segment, batch_size=batch_size)
        # Extract the transcribed text from the segments
        segment_text = " ".join([seg['text'] for seg in result["segments"]])
        extracted_text += " " + segment_text

    # Clear GPU memory after transcription
    torch.cuda.empty_cache()
    gc.collect()

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
