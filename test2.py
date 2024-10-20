import os
import torch

# Ensure the correct file path
url = 'process_text_audio_seperatly\\text_features_only'
file_name = 'text_features_and_labels.pt'

# Construct the absolute path
saved_data_path = os.path.join(os.getcwd(), url, file_name)

# Check if the file exists at the constructed path
if os.path.exists(saved_data_path):
    # Load the saved data
    loaded_data = torch.load(saved_data_path)

    # Extract features and labels
    features_list = loaded_data['features']
    labels = loaded_data['labels']

    # Print general statistics
    print("Number of samples:", len(features_list))

    # Example: Print the shape of the first few samples
    for i in range(10):  # Change the range if you want to print more samples
        print(f"Sample {i+1} - Features shape: {features_list[i].shape}, Label: {labels[i]}")

else:
    print(f"File not found at path: {saved_data_path}")
# # # import os

# # # folder_path = 'process_text_audio_seperatly\\text_features_only'
# # # file_name = 'text_features_and_labels.pt'
# # # file_path = os.path.join(folder_path, file_name)

# # # # Check if the folder exists
# # # if os.path.isdir(folder_path):
# # #     print(f"The folder '{folder_path}' exists.")
    
# # #     # Check if the file exists in the folder
# # #     if os.path.exists(file_path):
# # #         print(f"The file '{file_path}' exists.")
# # #     else:
# # #         print(f"The file '{file_path}' does not exist.")
# # # else:
# # #     print(f"The folder '{folder_path}' does not exist.")

