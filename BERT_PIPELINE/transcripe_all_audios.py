import os
import torch
import whisper
import json
import logging
from termcolor import colored

# Set up logging with colored outputs
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

def log_debug(message):
    logger.debug(colored(message, 'blue'))

def log_info(message):
    logger.info(colored(message, 'green'))

def log_warning(message):
    logger.warning(colored(message, 'yellow'))

def log_error(message):
    logger.error(colored(message, 'red'))

# Define the source folder for audio files
source_folder = r"D:\audios"

# Define the output file to save transcriptions
output_file = r"./transcriptions.jsonl"

# Define Whisper model and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
whisper_model = whisper.load_model("medium", device=device)

def transcribe_audio(audio_path):
    try:
        # Attempt transcription with GPU
        with torch.no_grad():
            result = whisper_model.transcribe(audio_path, language="en")
            return result["text"]
    except RuntimeError as e:
        if 'out of memory' in str(e):
            log_warning(f"Out of memory error on GPU for {audio_path}. Switching to CPU.")
            torch.cuda.empty_cache()

            # Retry transcription on CPU
            whisper_model.to("cpu")
            try:
                with torch.no_grad():
                    result = whisper_model.transcribe(audio_path, language="en")
                    return result["text"]
            finally:
                # Move model back to GPU for subsequent tasks
                whisper_model.to(device)
        else:
            log_error(f"Runtime error during transcription of {audio_path}: {e}")
            return None

def extract_label_from_filename(filename):
    possible_labels = ["A1", "A2", "B1_1", "B1_2", "B2", "C1", "C2"]
    for label in possible_labels:
        if label.lower() in filename.lower():
            return label
    return "Unknown"

def ask_user_for_label(filename):
    log_warning(f"Label for {filename} could not be determined. Please enter it manually:")
    valid_labels = ["A1", "A2", "B1_1", "B1_2", "B2", "C1", "C2"]
    while True:
        user_input = input(f"Enter label for {filename} (options: {', '.join(valid_labels)}): ").strip()
        if user_input in valid_labels:
            return user_input
        else:
            log_error("Invalid label entered. Please try again.")

def save_transcription_to_file(transcription_entry, output_path):
    try:
        with open(output_path, 'a', encoding='utf-8') as f:
            json.dump(transcription_entry, f)
            f.write("\n")
        log_info(f"Transcription saved: {transcription_entry['filename']} - Label: {transcription_entry['label']}")
    except Exception as e:
        log_error(f"Error saving transcription for {transcription_entry['filename']}: {e}")

def main():
    for idx, filename in enumerate(os.listdir(source_folder)):
        if filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg', '.wma')):
            audio_path = os.path.join(source_folder, filename)
            log_debug(f"Processing file {idx + 1}: {filename}")

            label = extract_label_from_filename(filename)
            if label == "Unknown":
                label = ask_user_for_label(filename)

            transcription_text = transcribe_audio(audio_path)

            if transcription_text:
                transcription_entry = {
                    "filename": filename,
                    "label": label,
                    "text": transcription_text
                }

                # Save the transcription directly to the file
                save_transcription_to_file(transcription_entry, output_file)

if __name__ == "__main__":
    main()