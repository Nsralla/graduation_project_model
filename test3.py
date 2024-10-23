import pandas as pd

# Data from the first text feature
first_group = [
    ("Sample 1", "torch.Size([1, 123, 1024])", 3),
    ("Sample 2", "torch.Size([1, 57, 1024])", 3),
    ("Sample 3", "torch.Size([1, 102, 1024])", 2),
    ("Sample 4", "torch.Size([1, 85, 1024])", 3),
    ("Sample 5", "torch.Size([1, 102, 1024])", 2),
    ("Sample 6", "torch.Size([1, 106, 1024])", 3),
    ("Sample 7", "torch.Size([1, 162, 1024])", 2),
    ("Sample 8", "torch.Size([1, 112, 1024])", 2),
    ("Sample 9", "torch.Size([1, 107, 1024])", 3),
    ("Sample 10", "torch.Size([1, 110, 1024])", 3),
    ("Sample 11", "torch.Size([1, 130, 1024])", 3),
    ("Sample 12", "torch.Size([1, 139, 1024])", 2),
    ("Sample 13", "torch.Size([1, 92, 1024])", 3),
    ("Sample 14", "torch.Size([1, 113, 1024])", 3),
    ("Sample 15", "torch.Size([1, 156, 1024])", 2),
]

# Data from the second group
second_group = [
    ("feature_0.pt", "torch.Size([1, 263, 1024])", 3),
    ("feature_1.pt", "torch.Size([1, 203, 1024])", 3),
    ("feature_2.pt", "torch.Size([1, 250, 1024])", 2),
    ("feature_3.pt", "torch.Size([1, 267, 1024])", 3),
    ("feature_4.pt", "torch.Size([1, 225, 1024])", 2),
    ("feature_5.pt", "torch.Size([1, 271, 1024])", 3),
    ("feature_6.pt", "torch.Size([1, 303, 1024])", 2),
    ("feature_7.pt", "torch.Size([1, 264, 1024])", 2),
    ("feature_8.pt", "torch.Size([1, 298, 1024])", 3),
    ("feature_9.pt", "torch.Size([1, 245, 1024])", 3),
    ("feature_10.pt", "torch.Size([1, 297, 1024])", 3),
    ("feature_11.pt", "torch.Size([1, 232, 1024])", 2),
    ("feature_12.pt", "torch.Size([1, 300, 1024])", 3),
    ("feature_13.pt", "torch.Size([1, 306, 1024])", 3),
    ("feature_14.pt", "torch.Size([1, 289, 1024])", 2),
]

# Combine the two groups into a single DataFrame
combined_data = []
for first, second in zip(first_group, second_group):
    combined_data.append((first[0], first[1], second[1], second[2]))

# Create a DataFrame and remove the "File" and "Label (Text)" columns
df = pd.DataFrame(combined_data, columns=["Sample", "Features Shape (Text)", "Features Shape (File)", "Label (File)"])

# Display the table in the console
print(df.to_string(index=False))
