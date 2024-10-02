
import torch

def mean_pooling_along_sequence(features, target_seq_len):
    """
    Apply mean pooling along the sequence length to reduce the number of feature vectors.
    
    Parameters:
        features (torch.Tensor): The input features of shape [batch_size, seq_len, d_model].
        target_seq_len (int): The target number of sequence steps (e.g., 79 for the text features).
    
    Returns:
        pooled_features (torch.Tensor): The features after applying mean pooling.
    """
    batch_size, seq_len, d_model = features.shape
    if seq_len == target_seq_len:
        return features
    
    # Calculate how many audio features should be pooled together to match the text sequence length
    pooling_factor = seq_len // target_seq_len  # Number of audio feature vectors to pool together
    remaining_features = seq_len % target_seq_len
    
    # Split the feature tensor into equally sized chunks for mean pooling
    pooled_features = []
    for i in range(target_seq_len):
        start_idx = i * pooling_factor
        end_idx = start_idx + pooling_factor
        if i < remaining_features:  # If there are remaining features, distribute them
            end_idx += 1
        pooled_segment = features[:, start_idx:end_idx, :].mean(dim=1)  # Apply mean pooling
        pooled_features.append(pooled_segment)
    
    pooled_features = torch.stack(pooled_features, dim=1)
    return pooled_features
