import os
import shutil
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter

# Function to extract label from filename
def extract_label_from_filename(filename, labels=['A1', 'C1', 'C2']):
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
data_dir = r'D:\Graduation_Project\Youtube Audios'
train_dir = r'Youtube audios categories\training youtube'
test_dir = r'Youtube audios categories\testing youtube'
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

# Function to count files in a folder
def count_files_in_folder(folder_path):
    return len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

# Create histograms of labels in each folder
def plot_label_histogram(files, folder_name):
    labels = [extract_label_from_filename(f) for f in files]
    labels = [label for label in labels if label is not None]
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

# Count files in each folder
main_folder_count = count_files_in_folder(data_dir)
train_folder_count = count_files_in_folder(train_dir)
test_folder_count = count_files_in_folder(test_dir)

# Verify and print counts
print(f"Total files in the main folder: {main_folder_count}")
print(f"Training files: {train_folder_count}")
print(f"Testing files: {test_folder_count}")

# Check if all files were processed
processed_files_count = train_folder_count + test_folder_count
if processed_files_count == len(all_files):
    print("All files have been successfully processed and moved to training and testing folders.")
else:
    print(f"Warning: Discrepancy detected! Processed files: {processed_files_count}, but expected: {len(all_files)}")
