# Audio processing + Wav2Vec + WhisperX + BERT
import os
import librosa
import noisereduce as nr
import soundfile as sf
import webrtcvad
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import whisperx

def reduce_noise_in_audio(input_file_path, output_file_path):
    """
    Function to reduce noise in an audio file using the noisereduce library,
    apply voice activity detection, normalize the audio, extract features using Wav2Vec 2.0,
    and transcribe audio using WhisperX.
    
    Parameters:
        input_file_path (str): Path to the input audio file.
        output_file_path (str): Path to save the output audio file with reduced noise.
    """
    # Check if the input file exists
    if not os.path.isfile(input_file_path):
        print(f"Error: The input file '{input_file_path}' does not exist.")
        return None

    # Load the audio file
    print(f"Loading audio file from: {input_file_path}")
    audio_data, sample_rate = librosa.load(input_file_path, sr=16000, mono=True)  # Ensure correct sample rate and mono
    print("Audio file loaded successfully.")

    # Perform noise reduction
    print("Reducing noise in audio...")
    reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=sample_rate)
    print("Noise reduction completed.")

    # Apply Voice Activity Detection (VAD)
    vad_audio = apply_vad(reduced_noise_audio, sample_rate)

    # Normalize the audio
    normalized_audio = normalize_audio(vad_audio)

    # Extract features using Wav2Vec 2.0
    wav2vec_features = extract_features_wav2vec2(normalized_audio, sample_rate)

    # Save the cleaned and normalized audio file
    sf.write(output_file_path, normalized_audio, sample_rate)
    print("Cleaned and normalized audio file saved successfully.")
    
    # Transcribe audio using WhisperX and return both text and wav2vec features
    transcribed_text = transcribe_audio_with_whisperx(output_file_path)
    
    return wav2vec_features, transcribed_text


def apply_vad(audio_data, sample_rate):
    """Apply Voice Activity Detection (VAD) to audio data."""
    print("Applying Voice Activity Detection (VAD)...")
    vad = webrtcvad.Vad()
    vad.set_mode(2)  # More aggressive mode to detect speech
    audio_pcm = (audio_data * 32768).astype(np.int16)
    frame_duration = 30  # ms
    frame_size = int(sample_rate * frame_duration / 1000)

    vad_audio = np.array([], dtype=np.int16)
    for i in range(0, len(audio_pcm) - frame_size, frame_size):
        frame = audio_pcm[i:i + frame_size]
        if vad.is_speech(frame.tobytes(), sample_rate):
            vad_audio = np.concatenate((vad_audio, frame))

    vad_audio_float = vad_audio.astype(np.float32) / 32768
    print("VAD applied successfully.")
    return vad_audio_float


def normalize_audio(audio_data, target_peak=0.95):
    """Normalize audio data to a consistent peak level."""
    peak = np.max(np.abs(audio_data))
    normalization_factor = target_peak / peak
    normalized_audio = audio_data * normalization_factor
    print("Audio normalized.")
    return normalized_audio


def extract_features_wav2vec2(audio_data, sample_rate):
    """
    Function to extract features from audio using Wav2Vec 2.0.
    
    Parameters:
        audio_data (numpy.ndarray): Normalized audio data.
        sample_rate (int): Sample rate of the audio data.
    
    Returns:
        torch.Tensor: Extracted features from Wav2Vec 2.0 model.
    """
    print("Extracting features using Wav2Vec 2.0...")

    # Load Wav2Vec 2.0 model and processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

    model.eval()  # Ensure model is in evaluation mode

    # Ensure the audio is in the correct format
    if not isinstance(audio_data, np.ndarray):
        raise ValueError("Audio data should be a numpy array.")
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    # Preprocess the audio data for the model with padding and attention mask
    print("Preprocessing audio data...")
    inputs = processor(
        audio_data, 
        sampling_rate=sample_rate, 
        return_tensors="pt", 
        padding=True,                # Enable padding to handle varying lengths
        return_attention_mask=True   # Generate attention masks for padded sequences
    )
    input_values = inputs.input_values.to(model.device)  # Ensure data is on the same device as the model
    attention_mask = inputs.attention_mask.to(model.device)  # Send attention mask to the same device
    
    print(f"Input values shape: {input_values.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    
    # Extract features
    print("Passing audio data through the Wav2Vec 2.0 model...")
    with torch.no_grad():
        outputs = model(input_values, attention_mask=attention_mask)
        features = outputs.last_hidden_state

    print("Feature extraction completed.")
    print(f"Extracted features shape: {features.shape}")

    return features

def transcribe_audio_with_whisperx(audio_file_path):
    """Transcribe audio using WhisperX and return transcribed text."""
    print(f"Transcribing audio using WhisperX: {audio_file_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisperx.load_model("large-v2", device, compute_type="float32")
    audio = whisperx.load_audio(audio_file_path)
    result = model.transcribe(audio)
    
    segments = result["segments"]
    extracted_text = " ".join([segment['text'] for segment in segments])
    
    print("Transcription completed. Extracted text:")
    print(extracted_text)
    return extracted_text
