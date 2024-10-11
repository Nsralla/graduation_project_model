import noisereduce as nr
import librosa
import os
import webrtcvad
import numpy as np
import torch
import webrtcvad
import gc
from padding import  logger

def reduce_noise_in_audio(input_file_path):
    """
    Function to reduce noise in an audio file, apply VAD, normalize, and save it.
    Returns the normalized audio data.
    """
    # Check if the input file exists
    if not os.path.isfile(input_file_path):
        print(f"Error: The input file '{input_file_path}' does not exist.")
        return None

    # Load the audio file
    try:
        audio_data, sample_rate = librosa.load(input_file_path, sr=16000, mono=True)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None

    # Define the segment length (1 minute and 30 seconds)
    segment_length = 90  # seconds

    # Split the audio if it exceeds the segment length
    total_duration = librosa.get_duration(y=audio_data, sr=sample_rate)
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

    # Initialize an empty list to store the processed segments
    processed_segments = []

    # Process each segment (noise reduction, VAD, normalization)
    for segment in segments:
        # Perform noise reduction
        try:
            reduced_noise_audio = nr.reduce_noise(y=segment, sr=sample_rate)
        except Exception as e:
            print(f"Error during noise reduction: {e}")
            return None

        # Apply Voice Activity Detection (VAD)
        vad_audio = apply_vad(reduced_noise_audio, sample_rate)

        # Normalize the audio
        normalized_audio = normalize_audio(vad_audio)

        # Append the processed segment
        processed_segments.append(normalized_audio)

    # Concatenate all processed segments to form the final audio
    final_audio = np.concatenate(processed_segments)

    return final_audio

def apply_vad(audio_data, sample_rate):
    """
    Function to apply Voice Activity Detection (VAD) on audio data.
    """
    # Initialize VAD
    vad = webrtcvad.Vad()
    vad.set_mode(2)  # Set aggressiveness mode (0-3, higher is more aggressive)

    # Convert audio to 16-bit PCM format required by webrtcvad
    audio_pcm = (audio_data * 32768).astype(np.int16)

    # Split audio into 30ms frames (required by VAD)
    frame_duration = 30  # in milliseconds
    frame_size = int(sample_rate * frame_duration / 1000)  # Calculate frame size

    # Apply VAD to each frame and concatenate the results
    vad_audio = np.array([], dtype=np.int16)
    
    for i in range(0, len(audio_pcm), frame_size):
        frame = audio_pcm[i:i + frame_size]

        # Check if the frame size is incomplete (i.e., smaller than expected)
        if len(frame) < frame_size:
            break  # Ignore incomplete frame at the end
        
        # Check if the frame contains speech
        is_speech = vad.is_speech(frame.tobytes(), sample_rate)
        
        if is_speech:
            vad_audio = np.concatenate((vad_audio, frame))

    # Convert back to floating-point format (from PCM 16-bit format)
    vad_audio_float = vad_audio.astype(np.float32) / 32768
    return vad_audio_float


def normalize_audio(audio_data, target_peak=0.95):
    """
    Function to normalize audio data to a consistent peak level.
    """
    print("Normalizing audio...")

    # Find the current peak amplitude
    peak = np.max(np.abs(audio_data))
    if peak == 0:
        print("Warning: Audio data has zero amplitude.")
        normalization_factor = 1.0
    else:
        normalization_factor = target_peak / peak

    # Normalize the audio data
    normalized_audio = audio_data * normalization_factor
    print(f"Audio normalized to peak level {target_peak}.")

    return normalized_audio
def extract_features_wav2vec2(audio_data, sample_rate, processor, model):
    """
    Function to extract features from a single audio data array using Wav2Vec 2.0.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Ensure the audio is in the correct format
    if not isinstance(audio_data, np.ndarray):
        raise ValueError("Audio data should be a numpy array.")
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    if len(audio_data) == 0:
        raise ValueError("Audio data is empty.")

    # Define segment length for feature extraction (e.g., 30 seconds)
    segment_length = 30  # seconds
    sample_rate = 16000
    segment_samples = segment_length * sample_rate

    # Split the audio into smaller segments for feature extraction
    segments = [
        audio_data[i:i + segment_samples] for i in range(0, len(audio_data), segment_samples)
    ]

    features_list = []
    cls_token_list = []  # To store CLS tokens from each segment

    # Process each segment separately
    for segment in segments:
        # Preprocess the audio data
        inputs = processor(
            segment,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=False
        )

        # Move inputs to the correct device
        input_values = inputs.input_values.to(device)

        try:
            with torch.no_grad():
                outputs = model(input_values)
                features = outputs.last_hidden_state.cpu()  # Move features to CPU to save GPU memory
                logger.debug(f"AUDIO Extracted features shape: {features.shape}")
                logger.info("-----------------------------------------------------");

            # Extract [CLS] token (usually the first token)
            cls_token = features[:, 0, :]  # Shape: [batch_size, feature_dim]

        except Exception as e:
            logger.error(f"Error during feature extraction: {e}")
            continue
        
          # Collect CLS token from each segment
        cls_token_list.append(cls_token)
        # Collect features from each segment
        features_list.append(features)

        # Clear memory after processing each segment
        del inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()
    
    # If there are multiple CLS tokens, we average them to get a single one
    final_cls_token = torch.mean(torch.stack(cls_token_list), dim=0, keepdim=True)
    logger.info(f"Final CLS token shape: {final_cls_token.shape}")
    logger.info("-----------------------------------------------------");
        
    # Use squeeze to remove the extra dimension
    final_cls_token = final_cls_token.squeeze(0)  # This will change the shape from [1, 1, 1024] to [1, 1024]

    # Concatenate features from all segments along the time dimension
    final_features = torch.cat(features_list, dim=1) if len(features_list) > 1 else features_list[0]

    return final_features, final_cls_token

def process_single_audio(input_file_path, processor, model):
    """
    Function to process a single audio file, reduce noise, and extract features using Wav2Vec2.
    """
    sample_rate = 16000  # Since we set sr=16000 when loading audio

    # Reduce noise in the audio file
    normalized_audio = reduce_noise_in_audio(input_file_path)
    if normalized_audio is None:
        print(f"Error processing file {input_file_path}. Normalized audio is None.")
        return None

    # Convert the normalized audio to a numpy array (required for extract_features_wav2vec2)
    if not isinstance(normalized_audio, np.ndarray):
        normalized_audio = np.array(normalized_audio, dtype=np.float32)

    # Extract features using Wav2Vec2
    features,  final_cls_token = extract_features_wav2vec2(normalized_audio, sample_rate, processor, model)

    return features, final_cls_token
