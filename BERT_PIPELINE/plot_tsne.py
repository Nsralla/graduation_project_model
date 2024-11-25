import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def extract_label_from_filename(filename):
    """
    Extracts the label from the filename.
    Assumes the label is the last two parts separated by underscores before the extension.
    Example: 'SM_CHN_PTJ1_002_B1_2.pt' -> 'B1_2'
    """
    base = os.path.splitext(filename)[0]
    parts = base.split('_')
    if len(parts) < 2:
        raise ValueError(f"Filename '{filename}' does not contain enough parts to extract a label.")
    label = '_'.join(parts[-2:])
    return label

def load_features_and_labels(feature_folder):
    """
    Loads features and labels from .pt files in the specified folder.
    
    Args:
        feature_folder (str): Path to the folder containing .pt feature files.
        
    Returns:
        features_array (np.ndarray): Array of pooled features.
        labels_ids (np.ndarray): Array of numerical label IDs.
        id_to_label (dict): Mapping from label IDs to label names.
    """
    feature_files = [os.path.join(feature_folder, f) for f in os.listdir(feature_folder) if f.endswith('.pt')]
    
    if not feature_files:
        raise FileNotFoundError(f"No .pt files found in the directory '{feature_folder}'.")
    
    features_list = []
    labels_list = []
    
    print("Loading features and extracting labels...")
    for feature_file in tqdm(feature_files, desc='Loading Features'):
        try:
            filename = os.path.basename(feature_file)
            label = extract_label_from_filename(filename)
            
            # Load the feature tensor
            feature_tensor = torch.load(feature_file)
            
            # Check the dimensions of the feature tensor
            if feature_tensor.ndim == 2:
                # Feature shape: [seq_len, hidden_size] -> Pool over seq_len
                feature_pooled = feature_tensor.mean(dim=0).numpy()
            elif feature_tensor.ndim == 1:
                # Feature shape: [hidden_size] -> Already pooled
                feature_pooled = feature_tensor.numpy()
            else:
                print(f"Unexpected feature shape {feature_tensor.shape} in file '{filename}'. Skipping.")
                continue
            
            features_list.append(feature_pooled)
            labels_list.append(label)
        
        except Exception as e:
            print(f"Error processing file '{feature_file}': {e}")
            continue
    
    if not features_list:
        raise ValueError("No features were loaded. Please check your feature files and labels.")
    
    # Convert lists to NumPy arrays
    try:
        features_array = np.vstack(features_list)  # Shape: [num_samples, feature_dim]
    except ValueError as ve:
        raise ValueError(f"Error stacking features: {ve}. Ensure all features have the same dimensions.")
    
    labels_array = np.array(labels_list)
    
    # Handle label encoding
    unique_labels = sorted(set(labels_array))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    labels_ids = np.array([label_to_id[label] for label in labels_array])
    
    return features_array, labels_ids, id_to_label

def apply_tsne(features, random_state=42):
    """
    Applies t-SNE to reduce feature dimensions to 2D.
    
    Args:
        features (np.ndarray): Array of features.
        random_state (int): Random state for reproducibility.
        
    Returns:
        features_2d (np.ndarray): 2D t-SNE transformed features.
    """
    print("Applying t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(features)
    return features_2d

def plot_tsne(features_2d, labels_ids, id_to_label, save_path=None):
    """
    Plots the t-SNE reduced features with labels.
    
    Args:
        features_2d (np.ndarray): 2D t-SNE features.
        labels_ids (np.ndarray): Numerical label IDs.
        id_to_label (dict): Mapping from label IDs to label names.
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    plt.figure(figsize=(12, 8))
    unique_labels = np.unique(labels_ids)
    
    # Define a color map
    cmap = plt.get_cmap('tab10')
    colors = cmap.colors[:len(unique_labels)]
    
    for idx, label_id in enumerate(unique_labels):
        indices = labels_ids == label_id
        plt.scatter(
            features_2d[indices, 0],
            features_2d[indices, 1],
            label=id_to_label[label_id],
            alpha=0.7,
            color=colors[idx]
        )
    
    plt.legend(title='Labels', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('t-SNE Visualization of BERT Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"t-SNE plot saved to '{save_path}'.")
    else:
        plt.show()

def main():
    # Specify the path to your feature files
    feature_folder = 'finetunned_testing_features_Icnale_base_model'  # Update this path as needed
    
    # Load features and labels
    try:
        features, labels_ids, id_to_label = load_features_and_labels(feature_folder)
    except Exception as e:
        print(f"Error loading features and labels: {e}")
        return
    
    # Apply t-SNE
    features_2d = apply_tsne(features)
    
    # Plot t-SNE results
    plot_tsne(features_2d, labels_ids, id_to_label, save_path='bert_features_tsne.png')

if __name__ == '__main__':
    main()
