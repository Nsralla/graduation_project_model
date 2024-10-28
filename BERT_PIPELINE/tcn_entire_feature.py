import torch
# Load the saved all token features
all_token_features = torch.load('all_token_features1.pt')

# Flatten the features to separate each sample in the batch
flattened_all_token_features = [embedding for batch in all_token_features for embedding in batch]
print(f"Flattened all token features. Total samples: {len(flattened_all_token_features)}")
# print the shape of the first embeddings
print(flattened_all_token_features[123].shape)
print(flattened_all_token_features[1].shape)
print(flattened_all_token_features[2].shape)
print(flattened_all_token_features[3].shape)
print(flattened_all_token_features[4].shape)

