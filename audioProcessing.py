import librosa
import os
import numpy as np
import torch
import webrtcvad
from padding import logger

def load_audio(input_file_path, sample_rate=16000):
    """
    Load audio file without preprocessing.
    """
    if not os.path.isfile(input_file_path):
        print(f"Error: The input file '{input_file_path}' does not exist.")
        return None

    try:
        audio_data, _ = librosa.load(input_file_path, sr=sample_rate, mono=True)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None

    return audio_data

def apply_vad(audio_data, sample_rate, frame_duration_ms=30):
    """
    Apply Voice Activity Detection (VAD) to filter out non-speech audio.
    """
    vad = webrtcvad.Vad()
    vad.set_mode(3)  # Mode 3 is the most aggressive mode for detecting speech

    # Convert audio to 16-bit PCM format (required by webrtcvad)
    audio_data_int16 = (audio_data * 32768).astype(np.int16)

    frame_size = int(sample_rate * frame_duration_ms / 1000)
    speech_frames = []

    for start in range(0, len(audio_data_int16), frame_size):
        end = min(start + frame_size, len(audio_data_int16))
        frame = audio_data_int16[start:end].tobytes()

        if len(frame) < frame_size * 2:
            continue

        if vad.is_speech(frame, sample_rate):
            speech_frames.append(audio_data[start:end])

    # Concatenate all the detected speech frames
    if speech_frames:
        speech_audio = np.concatenate(speech_frames)
    else:
        speech_audio = np.array([])  # If no speech detected, return an empty array

    return speech_audio

def process_single_audio(input_file_path, processor, model):
    """
    Process a single audio file and extract features using Wav2Vec2.
    """
    sample_rate = 16000

    audio_data = load_audio(input_file_path, sample_rate)
    if audio_data is None:
        print(f"Error processing file {input_file_path}. Audio data is None.")
        return None

    # Apply VAD to keep only speech segments
    speech_audio = apply_vad(audio_data, sample_rate)
    
    # Log the length of the audio after VAD
    vad_duration = len(speech_audio) / sample_rate
    logger.info(f"Length of audio after VAD: {vad_duration:.2f} seconds")
    if len(speech_audio) == 0:
        print(f"No speech detected in the file {input_file_path}.")
        return None

    # Extract features using Wav2Vec2
    features, final_cls_token = extract_features_wav2vec2(speech_audio, sample_rate, processor, model)

    return features, final_cls_token

def extract_features_wav2vec2(audio_data, sample_rate, processor, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not isinstance(audio_data, np.ndarray):
        raise ValueError("Audio data should be a numpy array.")
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    if len(audio_data) == 0:
        raise ValueError("Audio data is empty.")

    segment_length = 30  # seconds
    segment_samples = segment_length * sample_rate

    segments = [
        audio_data[i:i + segment_samples] for i in range(0, len(audio_data), segment_samples)
    ]

    cls_token_list = []
    features_list = []

    for segment in segments:
        inputs = processor(
            segment,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=False
        )

        input_values = inputs.input_values.to(device)

        try:
            with torch.no_grad():
                outputs = model(input_values)
                features = outputs.last_hidden_state

            cls_token = features[:, 0, :]

        except Exception as e:
            logger.error(f"Error during feature extraction: {e}")
            continue

        cls_token_list.append(cls_token)
        features_list.append(features)

    final_cls_token = torch.mean(torch.stack(cls_token_list), dim=0, keepdim=True)
    logger.info(f"Final CLS token shape: {final_cls_token.shape}")

    final_cls_token = final_cls_token.squeeze(0).unsqueeze(0)

    final_features = torch.cat(features_list, dim=1) if len(features_list) > 1 else features_list[0]

    final_cls_token = final_cls_token.cpu()
    final_features = final_features.cpu()

    torch.cuda.empty_cache()

    return final_features, final_cls_token.squeeze(0)
