import os
import re
from collections import Counter

def count_classes_in_folder(folder_path):
    # Define a pattern to match class labels (e.g., a1, b2, c2) in both old and new formats
    pattern = re.compile(r'_([A-Ca-c]\d)', re.IGNORECASE)
    
    # Initialize a counter
    class_counter = Counter()
    
    # Loop through the files in the specified folder
    for filename in os.listdir(folder_path):
        # Check if the file is an MP3 file
        if filename.endswith(".mp3"):
            # Search for the class label in the filename
            match = pattern.search(filename)
            if match:
                # Extract the class label and convert to uppercase for consistency
                class_label = match.group(1).upper()
                # Update the counter
                class_counter[class_label] += 1
    
    return class_counter

# Example usage
folder_path = "D:\\IELTS_audios\\channel1"
class_counts = count_classes_in_folder(folder_path)
print("Class Counts:", class_counts)
print("Total:", sum(class_counts.values()))
