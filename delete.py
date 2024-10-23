import os

def delete_audios(audio_files, folder_path):
    # Loop through the provided list of audio files
    for audio_file in audio_files:
        # Construct the full file path
        file_path = os.path.join(folder_path, audio_file)
        
        # Check if the file exists and delete it
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted: {audio_file}")
            except Exception as e:
                print(f"Failed to delete {audio_file}: {e}")
        else:
            print(f"File not found: {audio_file}")

# List of audio files to delete based on your input
audio_files_to_delete = [
    "ult53_video_C1 (1)_1.mp3",
    "ult53_video_C1 (10)_1.mp3",
    "ult53_video_C1 (11)_1.mp3",
    "ult53_video_C1 (12)_1.mp3",
    "ult53_video_C1 (13)_1.mp3",
    "ult53_video_C1 (14)_1.mp3",
    "ult53_video_C1 (15)_1.mp3",
    "ult53_video_C1 (16)_1.mp3",
    "ult53_video_C1 (2)_1.mp3",
    "ult53_video_C1 (3)_1.mp3",
    "ult53_video_C1 (4)_1.mp3",
    "ult53_video_C1 (5)_1.mp3",
    "ult53_video_C1 (6)_1.mp3",
    "ult53_video_C1 (7)_1.mp3",
    "ult53_video_C1 (8)_1.mp3",
    "ult53_video_C1 (9)_1.mp3",
    "ult53_video_C1_1.mp3",
    "ult54_video_B2 (1).mp3"
]

# Example usage
folder_path = "D:\\IELTS_audios\\channel1"
delete_audios(audio_files_to_delete, folder_path)
