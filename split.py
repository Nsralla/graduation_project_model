import os
import shutil
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter

# Function to extract label from filename
def extract_label_from_filename(filename, labels=['A2', 'B1_1', 'B1_2', 'B2']):
    """
    Extracts the label from the filename based on predefined labels.

    Args:
        filename (str): The filename from which to extract the label.
        labels (list): A list of possible labels.

    Returns:
        str or None: The extracted label or None if not found.
    """
    base_filename = os.path.basename(filename).lower()
    base_filename = os.path.splitext(base_filename)[0]
    for label in labels:
        if label.lower() in base_filename:
            return label
    return None

# Set your dataset directory, train, and test directories
data_dir = 'D:\\Graduation_Project\\ICNALE'
train_dir = 'D:\\Graduation_Project\\training_icnale'
test_dir = 'D:\\Graduation_Project\\testing_icnale'
test_size = 0.2
random_seed = 42

# Create train and test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# List all files in the data directory and shuffle them
all_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
random.shuffle(all_files)

# Split the data
train_files, test_files = train_test_split(all_files, test_size=test_size, random_state=random_seed)

# Move files to the train and test directories
for file in train_files:
    shutil.copy(os.path.join(data_dir, file), os.path.join(train_dir, file))

for file in test_files:
    shutil.copy(os.path.join(data_dir, file), os.path.join(test_dir, file))

# Create histograms of labels in each folder
def plot_label_histogram(files, folder_name):
    labels = [extract_label_from_filename(f) for f in files]
    label_counts = Counter(labels)
    
    plt.figure()
    plt.bar(label_counts.keys(), label_counts.values())
    plt.title(f'Label Distribution in {folder_name} Folder')
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.show()

# Plot histograms for training and testing folders
plot_label_histogram(train_files, 'Training')
plot_label_histogram(test_files, 'Testing')

print(f"Training files: {len(train_files)}")
print(f"Testing files: {len(test_files)}")
