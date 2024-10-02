import os

directory = r"C:\Users\nsrha\OneDrive\Desktop\ICNALE"

def check_files(directory):
    non_mp3_files = []

    # Iterate through files in the directory
    for file_name in os.listdir(directory):
        # Build the complete file path
        file_path = os.path.join(directory, file_name)

        # Check if it's a file (not a directory) and if it doesn't end with .mp3
        if os.path.isfile(file_path) and not file_name.lower().endswith('.mp3'):
            non_mp3_files.append(file_name)

    # If non-mp3 files exist, print them
    if non_mp3_files:
        print("The following files are not .mp3:")
        for file in non_mp3_files:
            print(file)
    else:
        print("All files are .mp3.")

# Run the function
check_files(directory)
