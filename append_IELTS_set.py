import os
def get_IELTS_audio_files():
    audio_directory = r"audios\manually"
    audio_files = []
# Iterate through the directory and add .mp3 files to the list
    for file_name in os.listdir(audio_directory):
        if file_name.lower().endswith(".mp3"):
            audio_files.append(os.path.join(audio_directory, file_name))
    return audio_files