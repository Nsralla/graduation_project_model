import os
import json
import shutil
from tqdm import tqdm

def load_training_filenames(training_jsonl_path):
    """
    Loads filenames from the training JSONL file.

    Args:
        training_jsonl_path (str): Path to the training JSONL file.

    Returns:
        set: Set of filenames for the training set.
    """
    training_filenames = set()
    with open(training_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            filename = data.get("filename")
            if filename:
                # Save only the base name (e.g., "example.mp3") for easy matching
                training_filenames.add(os.path.splitext(filename)[0])
    return training_filenames

def split_features(features_dir, training_filenames, output_training_dir, output_testing_dir):
    """
    Splits the extracted features into training and testing directories based on filenames.

    Args:
        features_dir (str): Directory containing all extracted feature files.
        training_filenames (set): Set of filenames that belong to the training set.
        output_training_dir (str): Directory to save training features.
        output_testing_dir (str): Directory to save testing features.
    """
    # Ensure output directories exist
    os.makedirs(output_training_dir, exist_ok=True)
    os.makedirs(output_testing_dir, exist_ok=True)

    # Process each feature file
    for feature_file in tqdm(os.listdir(features_dir), desc="Splitting Features"):
        # Extract the base filename (e.g., "example" from "example.pt")
        base_filename = os.path.splitext(feature_file)[0]

        # Determine the target directory based on whether the file is in the training set
        source_path = os.path.join(features_dir, feature_file)
        if base_filename in training_filenames:
            target_path = os.path.join(output_training_dir, feature_file)
        else:
            target_path = os.path.join(output_testing_dir, feature_file)

        # Move the file
        shutil.move(source_path, target_path)

def main():
    # Define paths
    training_jsonl_path = 'Icnale_training_transcription.jsonl'  # Path to your training JSONL file
    features_dir = './extracted_features_from_ICNALE_base_model'  # Directory where individual feature files are saved
    output_training_dir = './training_features_Icnale_base_model'
    output_testing_dir = './testing_features_Icnale_base_model'

    # Load training filenames
    training_filenames = load_training_filenames(training_jsonl_path)
    print(f"Loaded {len(training_filenames)} training filenames.")

    # Split features based on filenames
    split_features(features_dir, training_filenames, output_training_dir, output_testing_dir)

    print("Feature splitting completed.")

if __name__ == "__main__":
    main()
