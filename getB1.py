import os
import random
import shutil

source_directory = r"C:\Users\nsrha\OneDrive\Desktop\ICNALE"
destination_directory = r"audios\manually"
score_b1_files = []

# Collect all files with B1 in their path
for file_name in os.listdir(source_directory):
    if "B2" in file_name.upper():
        score_b1_files.append(file_name)

# Select 500 random B1 files
selected_b1_files = random.sample(score_b1_files, 500)

# Create the destination directory if it doesn't exist
os.makedirs(destination_directory, exist_ok=True)

# Copy the selected B1 files to the destination directory
for file_name in selected_b1_files:
    source_path = os.path.join(source_directory, file_name)
    destination_path = os.path.join(destination_directory, file_name)
    shutil.copy(source_path, destination_path)

print(f"Copied {len(selected_b1_files)} B1 files to the balanced directory.")
