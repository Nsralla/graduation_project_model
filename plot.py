import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import torch
import os
import numpy as np
from torch.nn.utils.rnn import pad_sequence

# Set up for plotting
sns.set(style="whitegrid")

# Load the data
url = 'process_text_audio_seperatly\\text_features_only'
file_name = 'text_features_and_labels.pt'
saved_data_path = os.path.join(os.getcwd(), url, file_name)

class_labels = ['A1', 'A2', 'B1_1', 'B1_2', 'B2', 'C1', 'C2']
num_classes = len(class_labels)

def pad_features(features_list):
    # Flatten out to list of tensors without the batch dimension
    features = [feature.squeeze(0) for feature in features_list]
    # Pad sequences to make them of uniform length
    padded_features = pad_sequence(features, batch_first=True, padding_value=0)
    return padded_features

if os.path.exists(saved_data_path):
    loaded_data = torch.load(saved_data_path)
    features_list = loaded_data['features']
    labels = loaded_data['labels']

    # Ensure reproducibility
    torch.manual_seed(42)

    # Pad the features to make them uniform
    padded_features = pad_features(features_list)

    # Convert to numpy if features are consistent
    features_array = padded_features.numpy()
    labels_array = np.array(labels)

    # Split the data into training and validation sets
    N = len(features_array)
    indices = np.random.permutation(N)
    train_size = int(0.8 * N)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_features = features_array[train_indices]
    train_labels = labels_array[train_indices]

    # Visualize class distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x=train_labels)
    plt.title("Class Distribution in Training Set")
    plt.xlabel("Classes")
    plt.ylabel("Frequency")
    plt.xticks(range(len(class_labels)), class_labels)
    plt.show()

    # Calculate and Plot Raw Feature Variance
    raw_variances = np.var(train_features.reshape(-1, train_features.shape[-1]), axis=0)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(raw_variances)), raw_variances)
    plt.title("Raw Feature Variance Before Thresholding")
    plt.xlabel("Feature Index")
    plt.ylabel("Variance")
    plt.show()

    # Apply Variance Thresholding
    selector = VarianceThreshold(threshold=0.0)
    try:
        train_features_var_thresh = selector.fit_transform(train_features.reshape(-1, train_features.shape[-1]))
        # Check if variance threshold is applied
        selected_variances = selector.variances_
    except ValueError as e:
        print(f"Variance Thresholding error: {e}")
        train_features_var_thresh = train_features  # Use raw features if variance fails
        selected_variances = raw_variances  # Fallback to raw variance

    # Plot Feature Variance After Thresholding
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(selected_variances)), selected_variances)
    plt.title("Feature Variance After Variance Thresholding")
    plt.xlabel("Feature Index")
    plt.ylabel("Variance")
    plt.show()

    # PCA for 2D visualization
    pca = PCA(n_components=2)
    features_pca_2d = pca.fit_transform(train_features_var_thresh)

    # Standardize features
    scaler = StandardScaler()
    features_pca_2d = scaler.fit_transform(features_pca_2d)

    # Scatter plot of PCA-transformed features
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=features_pca_2d[:, 0], y=features_pca_2d[:, 1], hue=train_labels, palette="Set1", s=60)
    plt.title("2D PCA Visualization of Training Features")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(class_labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

else:
    print(f"File not found at path: {saved_data_path}")
