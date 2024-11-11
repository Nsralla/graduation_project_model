import os
import torch
import numpy as np
from torch.nn import AvgPool1d
from tqdm import tqdm

def average_pool_features(input_dir, output_dir, window_size_ms=100, stride_size_ms=40, feature_stride_ms=20):
    """
    Apply average pooling to Wav2Vec2 features in the specified directory.

    Args:
        input_dir (str): Path to the input features directory containing .npz files.
        output_dir (str): Path to the output directory where pooled features will be saved.
        window_size_ms (int): Pooling window size in milliseconds.
        stride_size_ms (int): Pooling stride size in milliseconds.
        feature_stride_ms (int): Stride between each feature vector in milliseconds (default 20 ms).
    """
    # Calculate window and stride sizes in number of feature vectors
    window_size = window_size_ms // feature_stride_ms
    stride_size = stride_size_ms // feature_stride_ms

    print(f"\n--- Processing Directory: {input_dir} ---")
    print(f"Pooling Window Size: {window_size} feature vectors ({window_size_ms} ms)")
    print(f"Pooling Stride Size: {stride_size} feature vectors ({stride_size_ms} ms)\n")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize average pooling layer
    avg_pool = AvgPool1d(kernel_size=window_size, stride=stride_size)

    # Iterate over all .npz files in input_dir
    for file in tqdm(os.listdir(input_dir), desc=f'Processing {os.path.basename(input_dir)}'):
        if file.endswith('.npz'):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file)

            # Load the feature array and filename
            try:
                data = np.load(input_path)
                feature = data['feature']  # Shape: [seq_len, hidden_size]
                filename = str(data['filename'])  # Convert to string
            except Exception as e:
                print(f"Error loading {input_path}: {e}")
                continue

            # Debug: print feature shape
            print(f"Processing {file}: Feature shape before pooling: {feature.shape}")

            # Convert to tensor and reshape for pooling
            feature_tensor = torch.tensor(feature, dtype=torch.float)  # [seq_len, hidden_size]
            feature_tensor = feature_tensor.permute(1, 0).unsqueeze(0)  # [1, hidden_size, seq_len]

            # Apply average pooling
            pooled_tensor = avg_pool(feature_tensor)  # [1, hidden_size, pooled_seq_len]

            # Permute back to [pooled_seq_len, hidden_size]
            pooled_feature = pooled_tensor.squeeze(0).permute(1, 0).numpy()  # [pooled_seq_len, hidden_size]

            # Debug: print pooled feature shape
            print(f"Pooled feature shape: {pooled_feature.shape}")

            # Save the pooled feature to a new .npz file
            try:
                np.savez_compressed(
                    output_path,
                    feature=pooled_feature,
                    filename=filename  # Keep the same filename
                )
            except Exception as e:
                print(f"Error saving {output_path}: {e}")
                continue

    print(f"\nFeature pooling completed. Pooled features saved to {output_dir}\n")

if __name__ == '__main__':
    # Define input directories
    training_input_dir = './ICNALE_features_training_dataset'  # Update path as necessary
    testing_input_dir = './ICNALE_features_testing_dataset'    # Update path as necessary

    # Define output directories
    training_output_dir = 'secondAveragingtry/ICNALE_features_training_pooled_dataset'
    testing_output_dir = 'secondAveragingtry/ICNALE_features_testing_pooled_dataset'

    # Define pooling parameters (in milliseconds)
    window_size_ms = 200  # Example: 100 ms window
    stride_size_ms = 100   # Example: 40 ms stride

    # Pooling for training features
    print("Starting average pooling for training features...")
    average_pool_features(training_input_dir, training_output_dir, window_size_ms, stride_size_ms)

    # Pooling for testing features
    print("Starting average pooling for testing features...")
    average_pool_features(testing_input_dir, testing_output_dir, window_size_ms, stride_size_ms)

    print("All feature pooling operations completed successfully.\n")
