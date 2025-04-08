from abc import ABC, abstractmethod
from typing import List
import os
import pyarrow.parquet as pq
import pyarrow as pa

import pandas as pd

class BaseDataset(ABC):
    def __init__(self, name):
        """
        Initializes the Dataset object.

        Args:
            name (str): Name of the dataset (e.g., "MeDAL").
            path (str, optional): Path to the dataset. Default is None.
        """
        self.name = name
        self.path = None
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.embedding_type = None
        self.embedding_model = None


    @abstractmethod
    def load_dataset(self):
        """
        Abstract method to load the dataset from a specified path.
        This method must be implemented by the subclass.
        """
        pass

    # @abstractmethod
    # def run_pipeline(self):
    #     """
    #     Executes pipeline for downloading, loading, pre-processing the dataset
    #     """
    #     pass

    @abstractmethod
    def preprocess(self):
        """
        Abstract method for dataset-specific preprocessing.
        This method must be implemented by the subclass.
        """
        pass

    @abstractmethod
    def tokenize(self, tokenizer_type, splits):
        """
        Abstract method to tokenize the text data.
        This method must be implemented by the subclass.
        """
        pass

    @abstractmethod
    def embed(self, embedding_type, splits):
        """
        Abstract method to vectorize the tokenized data.
        This method must be implemented by the subclass.
        """
        pass

    @abstractmethod
    def split_dataset(self):
        """
        Abstract method to split the data into training, validation, and test sets.
        This method must be implemented by the subclass.
        """
        pass

    def save_embeddings(self, split_embeddings: List, splits: List[str], model_name: str):
        """
        Saves embeddings for a split into a Parquet file.
        """
        if not os.path.exists('saved_embeddings'):
            os.makedirs('saved_embeddings')

        for embeddings, split in zip(split_embeddings, splits):
            file_name = f'saved_embeddings/{split}_{model_name}_embeddings.parquet'

            try:
                # Convert embeddings list to Pandas DataFrame
                df = pd.DataFrame(embeddings)

                # Convert to Apache Arrow table
                table = pa.Table.from_pandas(df)

                # Save as Parquet file with Snappy compression (efficient & fast)
                pq.write_table(table, file_name, compression='snappy')

                print(f'{split} embeddings saved successfully in {file_name}\n')

            except Exception as e:
                raise ValueError(f'Error in saving embeddings! {e}')