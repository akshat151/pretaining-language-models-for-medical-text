import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_SelfAttention(nn.Module):
    def __init__(
        self, 
        embedding_dim,
        embedding_model,
        lstm_hidden_dim, 
        lstm_units,
        num_classes, 
        num_attention_heads,
        dropout=0.5,
        pretrained_embeddings=None,
        create_embedding_layer=True,
        max_sequence_length=50
    ):
        super().__init__()
        self.create_embedding_layer = create_embedding_layer

        if self.create_embedding_layer:
            assert embedding_model is not None, "embedding_model must be provided if use_embedding_layer=True"
            self.embedding = nn.Embedding(
                num_embeddings=len(embedding_model.embedding_matrix),
                embedding_dim=embedding_dim,
                padding_idx=0
            )
            self.embedding.weight.data.copy_(torch.tensor(embedding_model.embedding_matrix))
            self.embedding.weight.requires_grad = False  # Set to True if you wish to fine-tune the embeddings
        else:
            self.embedding = None

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_units,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # Attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=2 * lstm_hidden_dim, 
            num_heads=num_attention_heads,
            batch_first=True
        )

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(2 * lstm_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc1 = nn.Linear(4 * lstm_hidden_dim, num_classes)
        self.fc_embedding = nn.Linear(embedding_dim, 2 * lstm_hidden_dim)


    def forward(self, x, mask=None):
        if self.create_embedding_layer:
            x = x.long()
            x = self.embedding(x)  # (B, seq_len, embedding_dim)
        else:
            assert len(x.shape) == 3, "Expected input shape (B, seq_len, embedding_dim) if not using embedding layer"

        # Apply a linear transformation to x to match the shape of lstm_out (B, seq_len, 2 * lstm_hidden_dim)
        x_residual = self.dropout(self.fc_embedding(x))  # (B, seq_len, 2 * lstm_hidden_dim)
        if torch.isnan(x).any():
            print(f"[NaN DETECTED] NaN found in input tensor x. Shape: {x.shape}")
            return
        
        if torch.isnan(x_residual).any():
            print(x.shape)
            print(f"[NaN DETECTED] NaN found in input tensor x_residual. Shape: {x.shape}")
            return

        # LSTM output
        lstm_out, (h_n, _) = self.lstm(x)
        
        # Residual connection in LSTM output
        lstm_out_residual = lstm_out + x_residual  # Adding the transformed input to the LSTM output

        # Getting final LSTM hidden states
        forward_h = h_n[-2, :, :]
        backward_h = h_n[-1, :, :]
        final_lstm_hidden = torch.cat((forward_h, backward_h), dim=1)

        # Attention mechanism
        attn_mask = ~mask.bool() if mask is not None else None
        attention_out, _ = self.attention(lstm_out_residual, lstm_out_residual, lstm_out_residual, key_padding_mask=attn_mask)

        # Residual connection in attention output
        attention_out_residual = attention_out + lstm_out_residual  # Adding the residual from the LSTM output
        
        attention_out_residual = self.layer_norm(attention_out_residual)
        attention_summary = torch.mean(attention_out_residual, dim=1)

        # Combine the outputs
        combined_out = torch.cat((final_lstm_hidden, attention_summary), dim=1)

        # Apply dropout
        x = self.dropout(combined_out)

        # Fully connected layer
        x = self.fc1(x)

        return x


    def unfreeze_embeddings(self):
        if self.create_embedding_layer:
            self.embedding.weight.requires_grad = True
