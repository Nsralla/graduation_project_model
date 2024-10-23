import os
import hashlib
import whisper
import torch
from transformers import  XLNetTokenizer, XLNetModel
from padding import logger  # Assumed to be available based on your original code
# save the label
def compute_file_hash(file_path):
    hash_func = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def append_tensor_to_file(tensor, file_path):
    # Check if the file already exists
    if os.path.exists(file_path):
        # Load the existing tensor
        existing_tensor = torch.load(file_path)
        # Concatenate the new tensor with the existing one along the appropriate dimension
        combined_tensor = torch.cat((existing_tensor, tensor.cpu()), dim=0)
    else:
        # If the file does not exist, use the new tensor as the combined tensor
        combined_tensor = tensor.cpu()

    # Save the combined tensor back to the file
    torch.save(combined_tensor, file_path)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Whisper model on GPU
    whisper_model = whisper.load_model("medium", device="cuda")

  # Load pre-trained XLNet model and tokenizer
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
    bert_model = XLNetModel.from_pretrained('xlnet-large-cased').to(device)


    folder_path = "D:\\IELTS_audios\\channel1"

    # Supported audio formats
    audio_extensions = (".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".wma")

    # Set to store hashes of processed files
    processed_hashes = set()

    # Create directories for CLS tokens and text embeddings if they don't exist
    cls_token_dir = 'cls_text_tokens/'
    text_features_dir = 'text_features/'
    os.makedirs(cls_token_dir, exist_ok=True)
    os.makedirs(text_features_dir, exist_ok=True)

    # Iterate over all files in the folder
    for idx, filename in enumerate(os.listdir(folder_path)):
        logger.debug("--------------------------------------------------")
        if filename.endswith(audio_extensions):
            audio_path = os.path.join(folder_path, filename)

            # Compute hash of the audio file
            file_hash = compute_file_hash(audio_path)

            # Check if the file has already been processed
            if file_hash in processed_hashes:
                logger.info(f"deleint duplicate file {filename}")
                os.remove(audio_path)
                continue

            processed_hashes.add(file_hash)

            logger.info(f"Processing file {idx}: {filename}")

            try:
                # Monitor memory before transcription
                logger.info(f"Memory Allocated before: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

                # Transcribe the audio
                with torch.no_grad():
                    result = whisper_model.transcribe(audio_path,language="en")
                    text = result["text"]
                logger.info(f"Transcription for {filename}: {text}")

                # Tokenize input
                encoded_input = tokenizer(text, return_tensors='pt')

                # Move model and inputs to GPU if available
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                bert_model.to(device)
                encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

                # Get BERT embeddings
                with torch.no_grad():
                    outputs = bert_model(**encoded_input)
                    embeddings = outputs.last_hidden_state
                logger.debug(f"Embeddings shape: {embeddings.shape}")
                # Extract CLS token embedding (first token)
                cls_embedding = embeddings[:, 0, :]  # Shape: [1, hidden_size]
                logger.debug(f"CLS token embedding shape: {cls_embedding.shape}")

                # Save CLS token embedding (appending)
                cls_file_path = os.path.join(cls_token_dir, "cls_tokens.pt")
                append_tensor_to_file(cls_embedding, cls_file_path)

                # Save raw embeddings without flattening (appending)
                embeddings_file_path = os.path.join(text_features_dir, "text_features.pt")
                append_tensor_to_file(embeddings.squeeze(0), embeddings_file_path)

                logger.info(f"Finished processing {filename}")

                # Delete variables and free up GPU memory
                del result, text, encoded_input, outputs, embeddings, cls_embedding
                torch.cuda.empty_cache()

                # Monitor memory after processing
                logger.info(f"Memory Allocated after: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
                logger.debug("--------------------------------------------------")

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    logger.error(f"Out of memory error occurred at file {filename}, index {idx}.")
                    try:
                        # Free up GPU memory and move computations to CPU
                        torch.cuda.empty_cache()
                        device = torch.device('cpu')
                        bert_model.to(device)
                        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

                        # Try again on CPU
                        with torch.no_grad():
                            outputs = bert_model(**encoded_input)
                            embeddings = outputs.last_hidden_state

                        # Extract CLS token embedding (first token)
                        cls_embedding = embeddings[:, 0, :]

                        # Save embeddings as before
                        append_tensor_to_file(cls_embedding, cls_file_path)
                        append_tensor_to_file(embeddings.squeeze(0), embeddings_file_path)

                        logger.info(f"Finished processing {filename} on CPU")

                        # Delete variables and free up memory
                        del result, text, encoded_input, outputs, embeddings, cls_embedding
                        torch.cuda.empty_cache()

                    except Exception as e:
                        logger.error(f"Failed to process {filename} on CPU. Error: {e}")
                        logger.error(f"Stopping execution at file {filename}, index {idx}.")
                        break

                else:
                    logger.error(f"Runtime error at file {filename}, index {idx}. Error: {e}")
                    break

            except Exception as e:
                logger.error(f"Error processing {filename} at index {idx}: {e}")
                break

if __name__ == '__main__':
    main()
