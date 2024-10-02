import os

directory = r"C:\Users\nsrha\OneDrive\Desktop\ICNALE"

def count_all_files(directory):
    file_count = 0

    # Iterate through files in the directory
    for file_name in os.listdir(directory):
        # Check if it's a file (not a directory)
        if os.path.isfile(os.path.join(directory, file_name)):
            file_count += 1

    print(f"Total number of files: {file_count}")

# Run the function
count_all_files(directory)
