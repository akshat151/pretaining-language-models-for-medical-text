import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from typing import List, Tuple



class BioBERTModel(nn.Module):
    def __init__(self, model_name, output_dim=256, device='cuda'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.hidden_size = self.model.config.hidden_size
        self.output_dim = output_dim

        # Projection layer: (768 → output_dim)
        self.projection = nn.Linear(self.hidden_size, self.output_dim).to(self.device)

    def embed(self, texts: List[str], max_length=128) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state.squeeze(0)  # (B, seq_len, 768)
            projected_embeddings = self.projection(token_embeddings)  # (B, seq_len, output_dim)

        return projected_embeddings, inputs['attention_mask']
