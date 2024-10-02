import torch.nn as nn
import torch
class CrossModalAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        """
        CrossModalAttention module that applies multi-head attention to align audio and text features.
        
        Parameters:
            d_model (int): The dimension of the feature vectors (e.g., 768 for both Wav2Vec2 and BERT).
            n_heads (int): The number of attention heads for multi-head attention.
        """
        super(CrossModalAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)  # Layer normalization for stability
        self.fc = nn.Linear(d_model, d_model)  # Optional fully connected layer for further processing

    def forward(self, audio_features, text_features):
        """
        Forward pass to align audio and text features using cross-attention.
        
        Parameters:
            audio_features (torch.Tensor): Audio features of shape (batch_size, seq_len_a, d_model).
            text_features (torch.Tensor): Text features of shape (batch_size, seq_len_t, d_model).
        
        Returns:
            aligned_audio (torch.Tensor): Audio features aligned based on attention with text.
            aligned_text (torch.Tensor): Text features aligned based on attention with audio.
        """
        # Apply multi-head attention, allowing audio to attend to text, and text to attend to audio
        # Attention is applied in both directions
        attn_output_audio, _ = self.multihead_attn(audio_features, text_features, text_features)  # Align audio to text
        attn_output_text, _ = self.multihead_attn(text_features, audio_features, audio_features)  # Align text to audio

        # Optional: apply layer normalization and a linear transformation
        aligned_audio = self.layer_norm(attn_output_audio + audio_features)  # Residual connection with layer norm
        aligned_text = self.layer_norm(attn_output_text + text_features)

        # Optional: apply fully connected layer for further processing
        aligned_audio = self.fc(aligned_audio)
        aligned_text = self.fc(aligned_text)

        return aligned_audio, aligned_text

# Example Usage
def apply_attention(audio_features, text_features, d_model=768, n_heads=8):
    """
    Function to apply cross-modal attention to align audio and text features.
    
    Parameters:
        audio_features (torch.Tensor): Audio features extracted from Wav2Vec (shape: [batch_size, seq_len_a, d_model]).
        text_features (torch.Tensor): Text features extracted from BERT (shape: [batch_size, seq_len_t, d_model]).
        d_model (int): The dimension of the feature vectors (default=768).
        n_heads (int): Number of attention heads for the multi-head attention mechanism (default=8).
    
    Returns:
        aligned_audio (torch.Tensor): Audio features aligned with text features.
        aligned_text (torch.Tensor): Text features aligned with audio features.
    """
    cross_modal_attention = CrossModalAttention(d_model, n_heads).to(audio_features.device)
    aligned_audio, aligned_text = cross_modal_attention(audio_features, text_features)
    
    return aligned_audio, aligned_text
