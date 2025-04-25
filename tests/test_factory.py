import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import torch.nn as nn
import torch
from src.models.factory import ModelFactory
from src.models.model_architectures.lstm_self_attention import LSTM_SelfAttention

def test_get_supported_model():
    model_name = 'lstm_and_self_attention'
    model = ModelFactory.get_model(
        model_name=model_name,
        embedding_dim=100,
        lstm_hidden_dim=64,
        num_classes=5
    )
    
    assert isinstance(model, nn.Module)
    assert isinstance(model, LSTM_SelfAttention)


def test_get_unsupported_model():
    with pytest.raises(ValueError, match="not supported currently"):
        ModelFactory.get_model('non_existent_model', input_dim=10)


def test_model_forward_pass():
    # Check basic forward compatibility for LSTM_SelfAttention
    model = ModelFactory.get_model(
        model_name='lstm_and_self_attention',
        embedding_dim=50,
        lstm_hidden_dim=32,
        num_classes=3
    )
    
    dummy_input = torch.randn(8, 20, 50)  # (batch_size, sequence_length, embedding_dim)
    dummy_mask = torch.ones(8, 20).bool()  # attention mask

    outputs = model(dummy_input, dummy_mask)
    assert outputs.shape == (8, 3)
