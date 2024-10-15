import torch
import os

# Path to the main file containing features and labels
url = 'process_text_audio_seperatly\\text_features_only'
file_name = 'text_features_and_labels.pt'
file_path = os.path.join(url, file_name)

# Load the original data
try:
    data = torch.load(file_path)
    features_list = data['features']
    labels = data['labels']

    # Ensure both features and labels have the same number of samples
    assert len(features_list) == len(labels), "Mismatch between the number of features and labels!"

    # Calculate the split point (halfway)
    split_idx = len(features_list) // 2

    # Split the features and labels into two parts
    part1_features = features_list[:split_idx]
    part1_labels = labels[:split_idx]

    part2_features = features_list[split_idx:]
    part2_labels = labels[split_idx:]

    # Create data dictionaries for each part
    data_part1 = {'features': part1_features, 'labels': part1_labels}
    data_part2 = {'features': part2_features, 'labels': part2_labels}

    # Save the two parts to separate files
    part1_file_path = os.path.join(url, 'text_features_and_labels_part1.pt')
    part2_file_path = os.path.join(url, 'text_features_and_labels_part2.pt')

    torch.save(data_part1, part1_file_path)
    torch.save(data_part2, part2_file_path)

    print(f"Successfully split the file into two parts:\n"
          f"Part 1 saved at: {part1_file_path}\n"
          f"Part 2 saved at: {part2_file_path}")

except FileNotFoundError:
    print(f"Error: The file {file_path} does not exist.")
except Exception as e:
    print(f"An error occurred: {e}")
