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
source_folder = r'Youtube audios categories\training youtube'  # Adjust as necessary
output_file = 'Youtube/training_transcription.jsonl'

# Load Whisper model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
whisper_model = whisper.load_model("medium", device=device)

# Load completed filenames from the existing output file
def load_completed_filenames(output_path):
    completed_files = set()
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                completed_files.add(entry["filename"])
    except FileNotFoundError:
        log_info("Output file not found, starting fresh.")
    return completed_files

completed_filenames = load_completed_filenames(output_file)

def transcribe_audio(audio_path):
    try:
        with torch.no_grad():
            result = whisper_model.transcribe(audio_path, language="en")
            return result["text"]
    except RuntimeError as e:
        if 'out of memory' in str(e):
            log_warning(f"Out of memory error on GPU for {audio_path}. Switching to CPU.")
            torch.cuda.empty_cache()
            whisper_model.to("cpu")
            try:
                with torch.no_grad():
                    result = whisper_model.transcribe(audio_path, language="en")
                    return result["text"]
            finally:
                whisper_model.to(device)
        else:
            log_error(f"Runtime error during transcription of {audio_path}: {e}")
            return None

def extract_label_from_filename(filename):
    possible_labels = ["A1", "C1", "C2"]
    for label in possible_labels:
        if label.lower() in filename.lower():
            return label
    return "Unknown"

def handle_unknown_label(filename):
    log_warning(f"Label for {filename} could not be determined. Assigning 'Unknown' by default.")
    return "Unknown"

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
        if filename not in completed_filenames and filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg', '.wma')):
            audio_path = os.path.join(source_folder, filename)
            log_debug(f"Processing file {idx + 1}: {filename}")

            label = extract_label_from_filename(filename)
            if label == "Unknown":
                label = handle_unknown_label(filename)

            transcription_text = transcribe_audio(audio_path)

            if transcription_text:
                transcription_entry = {
                    "filename": filename,
                    "label": label,
                    "text": transcription_text
                }

                save_transcription_to_file(transcription_entry, output_file)

if __name__ == "__main__":
    main()
