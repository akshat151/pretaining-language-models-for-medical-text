

import setup 

import pandas as pd
import pytest
from src.data.matc import MATC

@pytest.fixture
def dummy_matc_instance():
    # Create dummy train, test dataframes with required columns
    df_train = pd.DataFrame({
        'medical_abstract': ["This is a TEST abstract.", "Another Test Abstract!"],
        'condition_label': [1, 2]
    })
    df_test = pd.DataFrame({
        'medical_abstract': ["Test abstract for testing."],
        'condition_label': [1]
    })
    instance = MATC("dummy_matc")
    # Manually set the attributes to avoid loading/parquet issues
    instance.train_data = df_train.copy()
    instance.test_data = df_test.copy()
    # For split_dataset, we also need to set a dummy train_data which can be shuffled
    instance.val_data = pd.DataFrame()  # will be overwritten
    # Set conversion maps needed by _preprocess_split
    instance.idx_to_class = {0:"class0", 1:"class1", 2:"class2"}
    return instance

def test_split_dataset(dummy_matc_instance):
    train, val = dummy_matc_instance.split_dataset(train_size=0.5, shuffle=False)
    # In our dummy train_data we have 2 rows so train should have 1 row and val 1 row.
    assert len(train) == 1
    assert len(val) == 1

def test_getitem(dummy_matc_instance):
    # __getitem__ should return a row for valid index
    row = dummy_matc_instance.__getitem__(0)
    assert isinstance(row, pd.Series)
    # Test out-of-bound error.
    with pytest.raises(ValueError):
        dummy_matc_instance.__getitem__(100)

def test_preprocess(dummy_matc_instance):
    # Here we simply check that a DataFrame is returned.
    processed = dummy_matc_instance.preprocess(splits=['train'])
    assert isinstance(processed, pd.DataFrame)