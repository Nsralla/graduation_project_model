from transformers import BertTokenizer, BertModel
import torch
def extract_features_from_text(text):
    """Extract features from text using BERT."""
    print("Extracting features from text using BERT...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = bert_model(**inputs)
        features = outputs.last_hidden_state

    print(f"BERT extracted features shape: {features.shape}")
    return features
