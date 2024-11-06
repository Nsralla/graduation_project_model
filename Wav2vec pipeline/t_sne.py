import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Define path to the saved features directory
feature_dir = 'Second_try_more_freezing_layers\ICNALE_features_testing_dataset'  # Update to the path where features are stored

# Load features and labels from the .npz files
features = []
labels = []
filenames = []

print("Loading feature files from directory:", feature_dir)

# List of valid labels
valid_labels = {'A2', 'B1_1', 'B1_2', 'B2'}

for file in os.listdir(feature_dir):
    if file.endswith('.npz'):
        # Load the .npz file and display the filename
        file_path = os.path.join(feature_dir, file)
        print(f"Loading file: {file_path}")
        
        data = np.load(file_path)
        feature = data['feature']  # [seq_len, hidden_size]
        filename = str(data['filename'])  # Ensure filename is a string
        
        # Debug: print shapes and types
        print(f"Feature shape before averaging: {feature.shape}, Filename: {filename}")
        
        features.append(feature)
        filenames.append(filename)
        
        # Extract the label from the filename by looking for the second-to-last segment
        parts = filename.split('_')
        if len(parts) >= 2:
            label = f"{parts[-2]}_{parts[-1]}" if f"{parts[-2]}_{parts[-1]}" in valid_labels else parts[-2]
        else:
            label = parts[-1]  # Fallback in case of unexpected format
            
        if label in valid_labels:
            labels.append(label)
            print("Label extracted:", label)
            print("------------------------------------------------")
            
        else:
            print(f"Warning: Label '{label}' not recognized in filename '{filename}'")

print("Total files loaded:", len(features))

# Convert lists to arrays for t-SNE input
# We will take the mean across the sequence length dimension to reduce each feature to [hidden_size]
features = np.array([np.mean(f, axis=0) for f in features])  # Shape: [num_samples, hidden_size]
labels = np.array(labels)  # Shape: [num_samples]

# Debug: check shapes of the feature matrix and labels array
print("Shape of feature matrix after averaging:", features.shape)
print("Shape of labels array:", labels.shape)
print("Unique labels extracted:", np.unique(labels))

# Apply t-SNE
print("Applying t-SNE on extracted features...")
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features)

# Debug: confirm the shape of the t-SNE result
print("Shape of t-SNE transformed features:", features_2d.shape)

# Plot t-SNE results
plt.figure(figsize=(10, 8))
unique_labels = np.unique(labels)
for label in unique_labels:
    indices = labels == label
    plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label=label, alpha=0.6)

plt.legend(title="Labels")
plt.title("t-SNE of Wav2Vec2 Testing Data Features")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.grid(True)
plt.show()

print("t-SNE plot completed.")
