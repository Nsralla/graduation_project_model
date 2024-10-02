from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import torch
import numpy as np

def dtw_alignment(audio_features, text_features):
    """
    Apply Dynamic Time Warping (DTW) to align audio features with text features.
    
    Parameters:
        audio_features (torch.Tensor): Audio features extracted from Wav2Vec (shape: [seq_len_a, feature_dim]).
        text_features (torch.Tensor): Text features extracted from BERT (shape: [seq_len_b, feature_dim]).
    
    Returns:
        aligned_audio (torch.Tensor): Audio features aligned to text features.
        aligned_text (torch.Tensor): Text features aligned to audio features.
    """
    audio_features_np = audio_features.cpu().numpy()  # Convert to NumPy for DTW
    text_features_np = text_features.cpu().numpy()

    # Initialize lists to store the aligned features
    aligned_audio = []
    aligned_text = []

    # Apply fastdtw to find the optimal path
    distance, path = fastdtw(audio_features_np, text_features_np, dist=euclidean)

    # Iterate over the alignment path
    for i, j in path:
        aligned_audio.append(audio_features_np[i])  # Align audio feature at index i
        aligned_text.append(text_features_np[j])    # Align text feature at index j

    # Convert back to tensors
    aligned_audio = torch.tensor(np.array(aligned_audio)).to(audio_features.device)
    aligned_text = torch.tensor(np.array(aligned_text)).to(text_features.device)

    return aligned_audio, aligned_text
