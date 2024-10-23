import os
import re
import hashlib
import whisper
import torch
import json
from padding import logger

def compute_file_hash(file_path):
    hash_func = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def extract_label(filename):
    # Try to extract label using regex
    match = re.search(r'_([A-C][1-2])', filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    else:
        # Ask user to manually provide the label if not found
        manual_label = input(f"Label not found for file '{filename}'. Please enter the label (A1, A2, B1, etc.): ").strip().upper()
        # Validate the user input
        if re.match(r'^[A-C][1-2]$', manual_label):
            return manual_label
        else:
            print("Invalid label format entered. Skipping this file.")
            return None

def save_transcription_to_file(filename, text, label):
    # Save as JSON Line (one JSON object per line)
    data = {"label": label, "text": text}
    with open("transcriptions.jsonl", "a", encoding="utf-8") as f:
        json.dump(data, f)
        f.write("\n")

def main():
    whisper_model = whisper.load_model("medium", device="cuda")
    folder_path = "D:\\IELTS_audios\\channel1"
    audio_extensions = (".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".wma")
    processed_hashes = set()
    
    
    
       # Start from index 363 by slicing the list of files
    files = os.listdir(folder_path)
    for idx, filename in enumerate(files[301:], start=301):
        logger.info(f"Processing file {idx}: {filename}")
        
        if filename.endswith(audio_extensions):
            audio_path = os.path.join(folder_path, filename)
            file_hash = compute_file_hash(audio_path)
            
            if file_hash in processed_hashes:
                logger.info(f"Deleting duplicate file {filename}")
                os.remove(audio_path)
                continue
            
            processed_hashes.add(file_hash)

            label = extract_label(filename)
            if not label:
                logger.warning(f"Label not found for file {filename}, skipping.")
                continue

            try:
                logger.info(f"Memory Allocated before: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

                with torch.no_grad():
                    result = whisper_model.transcribe(audio_path, language="en")
                    text = result["text"]

                logger.info(f"Transcription for {filename}: {text}")
                save_transcription_to_file(filename, text, label)

                del result, text
                torch.cuda.empty_cache()
                logger.debug("--------------------------------------------------")

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    logger.error(f"Out of memory error occurred at file {filename}, index {idx}.")
                else:
                    logger.error(f"Runtime error at file {filename}, index {idx}. Error: {e}")
                    break
            except Exception as e:
                logger.error(f"Error processing {filename} at index {idx}: {e}")
                break
        else :
            logger.info(f"Skipping file {filename} as it is not an audio file.")
if __name__ == '__main__':
    main()
