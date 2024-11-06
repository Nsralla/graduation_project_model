import os

def count_features(folder_path):
    """
    Counts the number of feature files in a given folder.

    Args:
        folder_path (str): Path to the folder.

    Returns:
        int: Number of feature files in the folder.
    """
    try:
        # List all files in the folder
        files = os.listdir(folder_path)
        # Count only the files (ignore directories if any)
        feature_count = sum(1 for file in files if os.path.isfile(os.path.join(folder_path, file)))
        return feature_count
    except FileNotFoundError:
        print(f"Folder not found: {folder_path}")
        return 0

def main():
    # Paths to the folders
    training_folder = 'testing_features_Icnale_base_model'
    testing_folder = 'training_features_Icnale_base_model'

    # Count features in each folder
    training_count = count_features(training_folder)
    testing_count = count_features(testing_folder)

    # Print the results
    print(f"Number of features in {training_folder}: {training_count}")
    print(f"Number of features in {testing_folder}: {testing_count}")

if __name__ == "__main__":
    main()
