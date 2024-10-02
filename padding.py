
import torch
import logging
import colorlog

def collate_fn(batch):
    features, labels = zip(*batch)

    # Ensure all features have the same feature dimension by padding them manually
    max_feature_dim = max(f.size(1) for f in features)  # Find the maximum feature dimension in the batch
    max_sequence_length = max(f.size(0) for f in features)  # Find the maximum sequence length in the batch

    # Pad each feature tensor to have the same feature dimension and sequence length
    padded_features = []
    for f in features:
        # Pad feature dimension if necessary
        feature_dim_padding = max_feature_dim - f.size(1)
        sequence_length_padding = max_sequence_length - f.size(0)
        
        # Pad the sequence length dimension and feature dimension
        # Padding format: (left_pad, right_pad, top_pad, bottom_pad)
        padded_feature = torch.nn.functional.pad(f, (0, feature_dim_padding, 0, sequence_length_padding))
        padded_features.append(padded_feature)

    # Stack the padded features
    padded_features = torch.stack(padded_features, dim=0)  # Shape: [batch_size, max_sequence_length, max_feature_dim]

    # Transpose to match the expected shape for Conv1d: [batch_size, input_channels, sequence_length]
    padded_features = padded_features.permute(0, 2, 1)  # Now shape: [batch_size, max_feature_dim, max_sequence_length]

    # Create the mask (1 for actual data, 0 for padded values)
    mask = torch.zeros(padded_features.size(0), padded_features.size(2), dtype=torch.bool)  # Adjusted to match new sequence length
    for i, f in enumerate(features):
        mask[i, :f.size(0)] = 1  # Mark only the actual data as valid

    # Convert labels to tensor
    labels = torch.tensor(labels, dtype=torch.long)

    return padded_features, labels, mask




handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s: %(message)s",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
))

logger = logging.getLogger('IELTSLogger')
logger.setLevel(logging.DEBUG)  # Adjust as needed
logger.addHandler(handler)