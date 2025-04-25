import pandas as pd
import pytest
from src.data.medal import MeDALSubset

@pytest.fixture
def dummy_medal_instance():
    # Create a small dummy dataset with expected columns for MEDAL
    df = pd.DataFrame({
        'TEXT': ["This is a sample text.", "Another sample TEXT!"],
        'LABEL': ["label1", "label2"],
        'LOCATION': [2, 1]
    })
    instance = MeDALSubset("dummy_medal")
    # Directly assign the dummy data to all splits for testing
    instance.train_data = df.copy()
    instance.val_data = df.copy()
    instance.test_data = df.copy()
    return instance

def test_convert_class_to_idx(dummy_medal_instance):
    idx_map = dummy_medal_instance.convert_class_to_idx(dummy_medal_instance.train_data)
    assert isinstance(idx_map, dict)
    # There should be 2 classes since there are 2 labels
    assert len(idx_map) == 2

def test_getitem(dummy_medal_instance):
    row = dummy_medal_instance.__getitem__(0)
    assert isinstance(row, pd.Series)
    with pytest.raises(ValueError):
        dummy_medal_instance.__getitem__(100)

def test_preprocess(dummy_medal_instance):
    processed = dummy_medal_instance.preprocess(splits=['train'])
    # After preprocessing, the TEXT column should be a string
    assert processed['TEXT'].iloc[0] != ""