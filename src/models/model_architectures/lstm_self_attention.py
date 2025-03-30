import torch
import torch.nn as nn

class LSTM_SelfAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def forward(self, x):
        pass



class LSTMandSelfAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes, num_layers=1, dropout=0.1):
        super(LSTMandSelfAttention, self).__init__()
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        
        # Self-Attention Layer
        self.attention = nn.Linear(hidden_dim * 2, 1)  # BiLSTM doubles the hidden size
        self.softmax = nn.Softmax(dim=1)
        
        # Fully Connected Layer for classification
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # LSTM Output: (batch_size, seq_len, hidden_dim * 2)
        lstm_out, _ = self.lstm(x)

        # Self-Attention: Compute attention scores
        attn_scores = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attn_weights = self.softmax(attn_scores)  # Normalize attention scores

        # Apply attention weights: Weighted sum of LSTM outputs
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)  # (batch_size, hidden_dim * 2)

        # Classification
        output = self.fc(self.dropout(context_vector))  # (batch_size, num_classes)
        
        return output
