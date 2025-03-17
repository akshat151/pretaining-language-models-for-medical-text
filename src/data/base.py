from abc import ABC, abstractmethod

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
    def __len__(self):
        """
        Abstract method to calculate the length of the dataset.
        This method must be implemented by the subclass.
        """
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """
        Abstract method to return the sample at given index.
        This method must be implemented by the subclass.
        """
        pass

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
    def tokenize(self, tokenizer):
        """
        Abstract method to tokenize the text data.
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

    # @abstractmethod
    # def save_processed_data(self, path):
    #     """
    #     Abstract method to save the processed dataset to a file.
    #     This method must be implemented by the subclass.
    #     """
    #     pass

    # @abstractmethod
    # def get_data(self):
    #     """
    #     Abstract method to return the processed dataset (train, validation, test).
    #     This method must be implemented by the subclass.
    #     """
    #     pass