import os
import shutil

source_directory = r"C:\Users\nsrha\OneDrive\Desktop\ICNALE"
destination_directory = r"audios\manually"
b2_files = []

# Collect all files with B2 in their path
for file_name in os.listdir(source_directory):
    if "A2" in file_name.upper():
        b2_files.append(file_name)

# Create the destination directory if it doesn't exist
os.makedirs(destination_directory, exist_ok=True)

# Copy B2 files to the destination directory
for file_name in b2_files:
    source_path = os.path.join(source_directory, file_name)
    destination_path = os.path.join(destination_directory, file_name)
    shutil.copy(source_path, destination_path)

print(f"Copied {len(b2_files)} A2 files to the destination directory: {destination_directory}")
