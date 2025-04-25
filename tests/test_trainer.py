import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.models.trainer import ModelTrainer
from unittest.mock import MagicMock
import yaml
import os


@pytest.fixture
def dummy_config(tmp_path):
    config = {
        'training': {
            'epochs': 1,
            'hyperparameters': {
                'learning_rate': 1e-3,
                'gradient_accumulation_steps': 1,
                'warmup_steps': 0
            },
            'weight_decay': 0.01,
            'optimizer': 'adam',
            'create_embedding_layer': False
        },
        'model_names': ['mock_model'],
        'models': {
            'mock_model': {
                'hyperparameters': {
                    'lstm_hidden_dim': 32
                },
                'base_params': {}
            }
        },
        'datasets': {
            'medal': {
                'num_classes': 2,
                'loss_function': 'CrossEntropyLoss'
            }
        }
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    return str(config_file)


@pytest.fixture
def dummy_data():
    x = torch.randn(100, 10)       # Inputs
    mask = torch.ones(100, 10)     # Attention mask
    y = torch.randint(0, 2, (100,)) # Labels
    dataset = TensorDataset(x, mask, y)
    return DataLoader(dataset, batch_size=16)


@pytest.fixture
def dummy_model():
    class DummyModel(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.fc = nn.Linear(10, kwargs['num_classes'])

        def forward(self, x, mask):
            return self.fc(x)

        def unfreeze_embeddings(self):
            pass  # mock unfreeze

    return DummyModel


def test_config_loading(dummy_config):
    trainer = ModelTrainer(config_file=os.path.basename(dummy_config))
    assert trainer.num_epochs == 1


def test_training_pipeline(monkeypatch, dummy_config, dummy_data, dummy_model):
    trainer = ModelTrainer(config_file=os.path.basename(dummy_config))

    # Monkey patch the model factory
    trainer.model_factory.get_model = lambda name, **kwargs: dummy_model(num_classes=2)

    # Replace save_model to avoid writing to disk
    trainer.save_model = MagicMock()

    results = trainer.train(dummy_data, dummy_data, dataset='medal', embedding_dim=10)
    assert 'epoch_1' in results
    assert 'validation_1' in results


def test_evaluation(monkeypatch, dummy_config, dummy_data, dummy_model):
    trainer = ModelTrainer(config_file=os.path.basename(dummy_config))

    model = dummy_model(num_classes=2)
    model.eval()

    val_loss, val_acc, val_prec, val_recall = trainer.evaluate(dummy_data, model, dataset_name='medal')
    assert isinstance(val_loss, float)
    assert 0 <= val_acc <= 1
