import torch.nn as nn
import torch

class CTCAlignmentModel(nn.Module):
    def __init__(self, wav2vec_dim, bert_dim, hidden_dim, num_classes):
        super(CTCAlignmentModel, self).__init__()
        self.fc_wav2vec = nn.Linear(wav2vec_dim, hidden_dim)
        self.fc_bert = nn.Linear(bert_dim, hidden_dim)
        self.rnn = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_out = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, wav2vec_features, bert_features):
        # Project Wav2Vec and BERT features to a common hidden dimension
        wav2vec_proj = self.fc_wav2vec(wav2vec_features)
        bert_proj = self.fc_bert(bert_features)

        # Concatenate along the feature dimension
        combined_features = torch.cat((wav2vec_proj, bert_proj), dim=-1)

        # Pass through RNN
        rnn_out, _ = self.rnn(combined_features)

        # Output layer
        logits = self.fc_out(rnn_out)

        return logits