import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px

# Optional: For better plot aesthetics
sns.set(style='whitegrid', palette='muted', color_codes=True)

# ------------------------------
# 1. Load [CLS] Embeddings and Labels
# ------------------------------
cls_features_path = 'cls_features.pt'
labels_path = 'labels.pkl'

# Load [CLS] features
cls_features = torch.load(cls_features_path)  # List of tensors
cls_features_tensor = torch.stack(cls_features)  # Shape: [num_samples, hidden_size]
cls_features_np = cls_features_tensor.numpy()

print(f"Loaded [CLS] features shape: {cls_features_np.shape}")  # e.g., (1000, 768)

# Load labels
with open(labels_path, 'rb') as f:
    labels = pickle.load(f)  # List of labels

print(f"Number of labels: {len(labels)}")

# Verify the number of labels matches the number of embeddings
assert len(labels) == cls_features_np.shape[0], "Mismatch between labels and embeddings."

# ------------------------------
# 2. Preprocess the Embeddings
# ------------------------------
scaler = StandardScaler()
cls_features_scaled = scaler.fit_transform(cls_features_np)
print("Embeddings have been standardized.")

# ------------------------------
# 3. Dimensionality Reduction
# ------------------------------

# a. PCA
pca = PCA(n_components=2, random_state=42)
cls_features_pca = pca.fit_transform(cls_features_scaled)
print("PCA completed.")

# b. t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42, verbose=1)
cls_features_tsne = tsne.fit_transform(cls_features_scaled)
print("t-SNE completed.")

# ------------------------------
# 4. Visualization
# ------------------------------

# a. PCA Scatter Plot with Seaborn
df_pca = pd.DataFrame({
    'PC1': cls_features_pca[:, 0],
    'PC2': cls_features_pca[:, 1],
    'Label': labels
})

plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df_pca, x='PC1', y='PC2',
    hue='Label', palette='tab10',
    alpha=0.7, edgecolor='k', s=100
)
plt.title('PCA Projection of [CLS] Embeddings')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# b. t-SNE Scatter Plot with Seaborn
df_tsne = pd.DataFrame({
    'TSNE1': cls_features_tsne[:, 0],
    'TSNE2': cls_features_tsne[:, 1],
    'Label': labels
})

plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df_tsne, x='TSNE1', y='TSNE2',
    hue='Label', palette='tab10',
    alpha=0.7, edgecolor='k', s=100
)
plt.title('t-SNE Projection of [CLS] Embeddings')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# c. Interactive t-SNE Plot with Plotly
fig_tsne = px.scatter(
    df_tsne, x='TSNE1', y='TSNE2',
    color='Label', title='Interactive t-SNE Projection of [CLS] Embeddings',
    hover_data=['Label'],
    opacity=0.7
)
fig_tsne.show()

# d. Optional: PCA and t-SNE Side by Side
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# PCA Plot
sns.scatterplot(
    ax=axes[0],
    data=df_pca, x='PC1', y='PC2',
    hue='Label', palette='tab10',
    alpha=0.7, edgecolor='k', s=100
)
axes[0].set_title('PCA Projection of [CLS] Embeddings')
axes[0].set_xlabel('Principal Component 1')
axes[0].set_ylabel('Principal Component 2')
axes[0].legend(title='Label')

# t-SNE Plot
sns.scatterplot(
    ax=axes[1],
    data=df_tsne, x='TSNE1', y='TSNE2',
    hue='Label', palette='tab10',
    alpha=0.7, edgecolor='k', s=100
)
axes[1].set_title('t-SNE Projection of [CLS] Embeddings')
axes[1].set_xlabel('t-SNE Dimension 1')
axes[1].set_ylabel('t-SNE Dimension 2')
axes[1].legend(title='Label')

plt.tight_layout()
plt.show()
