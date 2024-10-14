import librosa
import os
import numpy as np
import torch
import webrtcvad
from padding import logger
import gc
from noisereduce import reduce_noise


def process_single_audio(input_file_path, processor, model):
    """
    Process a single audio file and extract features using Wav2Vec2.
    """
    sample_rate = 16000

    # load the audio
    audio_data = load_audio(input_file_path, sample_rate)
    if audio_data is None:
        print(f"Error processing file {input_file_path}. Audio data is None.")
        return None
    
    # reduce noise
    audio_data = enhance_audio(audio_data, sample_rate)

    # Apply VAD to keep only speech segments
    speech_audio = apply_vad(audio_data, sample_rate)
    
    # Log the length of the audio after VAD
    vad_duration = len(speech_audio) / sample_rate
    logger.info(f"Length of audio after VAD: {vad_duration:.2f} seconds")
    if len(speech_audio) == 0:
        print(f"No speech detected in the file {input_file_path}.")
        return None

    # Extract features using Wav2Vec2
    features= extract_features_wav2vec2(speech_audio, sample_rate, processor, model)

    return features

def load_audio(input_file_path, sample_rate=16000):
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
    vad.set_mode(2)  # Mode 3 is the most aggressive mode for detecting speech

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

def enhance_audio(audio_data, sample_rate):
    """
    Enhance the audio by reducing noise.
    """
    # Apply noise reduction
    reduced_noise_audio = reduce_noise(y=audio_data, sr=sample_rate)
    
    return reduced_noise_audio

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

    features_list = []

    for segment in segments:
        inputs = processor(
            segment,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=False
        )

        input_values = inputs.input_values.to(device)

        retries = 0
        max_retries = 3
        success = False

        while not success and retries < max_retries:
            try:
                with torch.no_grad():
                    outputs = model(input_values.to(device))  # Ensure inputs are moved to the right device
                    features = outputs.last_hidden_state
                    logger.info(f"Features shape: {features.shape}")
                    features_list.append(features)
                    success = True  # Mark as successful
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.warning(f"CUDA OOM error during feature extraction. Retrying... ({retries+1}/{max_retries})")
                    retries += 1
                    torch.cuda.empty_cache()
                    gc.collect()

                    # If retries are exhausted, fallback to CPU
                    if retries == max_retries:
                        logger.error("Max retries exceeded. Falling back to CPU.")
                        device = torch.device("cpu")
                        input_values = input_values.to(device)  # Move inputs to CPU
                        model.to(device)  # Move model to CPU
                else:
                    logger.error(f"Unexpected error during feature extraction: {e}")
                    raise
            except Exception as e:
                logger.error(f"Error during feature extraction: {e}")
                raise

    if len(features_list) > 1:
        final_features = torch.cat(features_list, dim=1)
    else:
        final_features = features_list[0]

    logger.info(f"Final features shape: {final_features.shape}")

    # Ensure the final features are moved to the CPU to free up GPU memory
    final_features = final_features.cpu()
    torch.cuda.empty_cache()
    gc.collect()

    return final_features