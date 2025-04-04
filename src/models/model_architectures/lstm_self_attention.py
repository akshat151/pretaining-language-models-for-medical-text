import torch
import torch.nn as nn
import torch.nn.functional as F

# Self-Attention Mechanism
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)  # Linear layer to compute attention score for each hidden state

    def forward(self, lstm_out):
        # lstm_out is of shape (batch_size, seq_len, hidden_dim)
        
        # Compute attention scores
        attn_scores = self.attn(lstm_out)  # (batch_size, seq_len, 1)
        
        # Normalize scores using softmax (along the sequence length)
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch_size, seq_len, 1)
        
        # Compute the weighted sum of the LSTM outputs (context vector)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)  # (batch_size, hidden_dim)
        
        return context_vector


class LSTM_SelfAttention(nn.Module):
    def __init__(self, embedding_dim, lstm_hidden_dim, lstm_units, num_classes, dropout=0.5):
        super().__init__()
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_dim, 
                            num_layers=lstm_units, batch_first=True, bidirectional=True)
        
        # Self-Attention Layer
        self.attention = SelfAttention(lstm_hidden_dim * 2)  # BiLSTM doubles the hidden size
        
        # Fully Connected Layer for classification
        self.fc = nn.Linear(lstm_hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # LSTM Output: (batch_size, seq_len, hidden_dim * 2)
        lstm_out, _ = self.lstm(x)
        
        # Apply Self-Attention: (batch_size, hidden_dim * 2)
        context_vector = self.attention(lstm_out)

        if mask is not None:
            context_vector = context_vector * mask
        
        # Classification using the context vector
        output = self.fc(self.dropout(context_vector))  # (batch_size, num_classes)
        
        return output
