import os
from moviepy.editor import *
from padding import logger

def convert_mp4_to_mp3(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            input_file = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]
            output_file = os.path.join(output_folder, f"{base_name}.mp3")

            # Generate a unique filename if the file already exists
            counter = 1
            while os.path.exists(output_file):
                output_file = os.path.join(output_folder, f"{base_name}_{counter}.mp3")
                counter += 1

            # Load the video file and extract audio
            try:
                video = VideoFileClip(input_file)
                audio_data = video.audio

                # Write audio to the output file
                audio_data.write_audiofile(output_file, codec="mp3", ffmpeg_params=["-f", "mp3"])
                video.close()
                logger.info(f"Converted {filename} to MP3 successfully as {os.path.basename(output_file)}.")
            except Exception as e:
                logger.error(f"Failed to convert {filename}: {e}")

# Example usage
input_folder = "C:\\Users\\nsrha\\Downloads\\mp4"
output_folder = "D:\\IELTS_audios\\channel1"
convert_mp4_to_mp3(input_folder, output_folder)
