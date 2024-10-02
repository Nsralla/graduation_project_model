# import os

# directory = r"C:\Users\nsrha\OneDrive\Desktop\ICNALE"

# def delete_files_with_xx(directory):
#     # Iterate through files in the directory
#     for file_name in os.listdir(directory):
#         # Check if the file contains "XX" and if it's a file
#         if "XX" in file_name and os.path.isfile(os.path.join(directory, file_name)):
#             file_path = os.path.join(directory, file_name)
#             # Delete the file
#             os.remove(file_path)
#             print(f"Deleted: {file_path}")

# # Run the function
# delete_files_with_xx(directory)
# import torch
# print(torch.version.cuda)  # Should now output '11.8'
# print(torch.cuda.is_available())  # Should return True if the GPU is available
