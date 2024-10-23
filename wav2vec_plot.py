import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm  # Optional, for displaying progress

# Define the directory containing the features
features_directory = os.path.join(os.getcwd(), 'intermediate_results', 'processed_features')

# Initialize lists to store features and labels
feature_list = []
label_list = []

# Optional: Define the maximum number of samples to process (for performance)
max_samples = 2930  # Set to an integer if you want to limit the number of samples

# Iterate over all .pt files in the directory
file_names = [f for f in os.listdir(features_directory) if f.endswith('.pt')]

# If you want to limit the number of samples
if max_samples is not None:
    file_names = file_names[:max_samples]

print("Loading features and labels...")
for file_name in tqdm(file_names):
    file_path = os.path.join(features_directory, file_name)
    data = torch.load(file_path)
    pooled_feature = data['pooled_feature']  # Shape: [1, sequence_length, 1024]
    label = data['label']  # Assuming label is a scalar or string

    # Compute the mean over the sequence length to obtain a fixed-size vector
    # pooled_feature has shape [1, sequence_length, 1024]
    # We'll average over the sequence_length dimension (dimension 1)
    feature_vector = pooled_feature.mean(dim=1).squeeze(0)  # Shape: [1024]

    # Convert to numpy array and append to list
    feature_list.append(feature_vector.numpy())
    label_list.append(label)

# Convert lists to arrays
features_array = np.stack(feature_list)  # Shape: [num_samples, 1024]
labels_array = np.array(label_list)

print("Features and labels loaded.")
print(f"Number of samples: {features_array.shape[0]}")
print(f"Feature vector size: {features_array.shape[1]}")

# Map labels to integers if they are not already numeric
# Create a label mapping
unique_labels = np.unique(labels_array)
label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
int_labels = np.array([label_to_int[label] for label in labels_array])

# Apply t-SNE to reduce dimensionality
print("Applying t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
features_2d = tsne.fit_transform(features_array)

print("t-SNE completed.")

# Plot the results
print("Visualizing embeddings...")
plt.figure(figsize=(12, 8))

# Define a color map
num_classes = len(unique_labels)
colors = plt.cm.get_cmap('tab10', num_classes)

for idx, label in enumerate(unique_labels):
    indices = np.where(labels_array == label)
    plt.scatter(features_2d[indices, 0], features_2d[indices, 1],
                color=colors(idx), label=str(label), alpha=0.6)

plt.legend(title='Labels')
plt.title('t-SNE Visualization of wav2vec Embeddings')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True)
plt.show()
