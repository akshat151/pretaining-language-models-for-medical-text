import shutil
from typing import List, Tuple, Union
import torch.nn as nn

from tokenizer.factory import TokenizerFactory
from ..models.embedding.pretrained_embedding import PretrainedEmbedding
from ..models.embedding.custom_embedding import CustomEmbedding

import spacy
from .base import BaseDataset
import kagglehub
import os
from env import ProjectPaths
from tqdm import tqdm
import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')

class MeDALSubset(BaseDataset):
    """
    Full MeDAL dataset is extremely large ~ 14GB.
    This class includes a subset of the MeDAL dataset ~ 5M entries.
    """
    def __init__(self, name):
        super().__init__(name)
        print(f'MeDAL dataset initialized with name: {self.name}')


    def _initialize_embedding_model(self) -> nn.Module:
        """
        Initializes the embedding model based on the embedding type.
        """
        if self.embedding_type == 'custom':
            vocab_size = 10000  # Example vocab size
            embedding_dim = 300  # Example embedding dimension
            return CustomEmbedding(vocab_size, embedding_dim)
        elif self.embedding_type == 'pretrained':
            return PretrainedEmbedding(model_name='bert-base-uncased')
        else:
            raise ValueError("Invalid embedding type. Choose between 'custom' or 'pretrained'.")



    def load_dataset(self) -> pd.DataFrame:
        """
        Loads the MeDAL dataset from Kaggle
        """
        
        downloaded_path = kagglehub.dataset_download("xhlulu/medal-emnlp")
        print("Dataset downloaded to:", downloaded_path)

        medal_dir = ProjectPaths.DATASET_DIR.value / 'medal'

        if not medal_dir.exists() or not medal_dir.is_dir():
            items = os.listdir(downloaded_path)

            for item in tqdm(items, desc='Moving dataset to project dir', unit='item'):
                item_path = os.path.join(downloaded_path, item)
                if os.path.isdir(item_path):
                    # Move folders
                    shutil.move(item_path, medal_dir / item)
                else:
                    # Move files
                    shutil.move(item_path, medal_dir)

        self.path = medal_dir / 'pretrain_subset' # Points to pretrain_subset

        self.train_data = pd.read_csv(self.path / 'train.csv')
        self.val_data = pd.read_csv(self.path / 'valid.csv')
        self.test_data = pd.read_csv(self.path / 'test.csv')
        self.data = pd.concat([self.train_data, self.val_data, self.test_data], ignore_index=True) 

        print(f"Dataset moved to: {ProjectPaths.DATASET_DIR.value}")
        return self.data

    @staticmethod
    def lemmatizer(data: str):
        # nlp = spacy.load('en_core_sci_md') # trained on general biomedical text data
        doc = nlp(data)
        lemmatized_data = ''
        for token in doc:
            lemmatized_data += f'{token.lemma_} ' 
        return lemmatized_data
    

    def split_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads the dataset into an `DataFrame` objects and parses it into train, val, test sets.

        Returns:
        train_data, val_data, test_data
        """
        return self.train_data, self.val_data, self.test_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if idx >= len(self.data) or idx < 0:
            raise ValueError('Index out of bounds')
        
        return self.data.iloc[idx]
    
    @staticmethod
    def remove_stop_words(data: str) -> str:
        stop_words = set(stopwords.words('english'))
        processed_data = ''
        for word in data.split():
            if word not in stop_words:
                processed_data += f'{word} '

        return processed_data
    

    def _preprocess_split(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Performs basic pre-processing on the dataset like replacing location
        of the word with the abbreviation, converting to lower case, removing
        stop words, etc.
        """

        # Rename the column LOCATION to ABBREVIATION for the entire DataFrame
        data = data.rename(columns={'LOCATION': 'ABBREVIATION'})

        for idx in range(len(data)):
            # Using .loc ensures you're modifying the actual DataFrame
            sample = data.loc[idx]  # Use .loc to access the row

            text: str = sample['TEXT']

            # Convert location to actual abbreviation. TODO: Should it be capital or small???
            # Leaving it as is for now
            words = text.split()
            abbreviation = words[sample['ABBREVIATION']]
            data.loc[idx, 'ABBREVIATION'] = abbreviation  # Modify the correct column

            # Convert to lower case
            text = text.lower()

            # Stemming, lemmatization, or other processing
            text = MeDALSubset.lemmatizer(text)

            # Remove stop words
            text = MeDALSubset.remove_stop_words(text)

            # Modify the actual 'TEXT' column in the DataFrame using .loc
            data.loc[idx, 'TEXT'] = text

        return data


    def preprocess(self, splits=['train']) -> Union[pd.DataFrame, 
                                                     Tuple[pd.DataFrame, 
                                                           pd.DataFrame], 
                                                           Tuple
                                                           [pd.DataFrame, 
                                                            pd.DataFrame, 
                                                            pd.DataFrame]]:
        """
        Performs basic pre-processing based on the splits passed.

        Parameters:
        splits (list): A list of dataset splits to pre-process. It can contain any of the following:
            'train', 'valid', 'test'. The specified splits will be pre-processed accordingly.
            By default, it processes the 'train' split.

        Example:
        - pre_process(['train', 'valid']) will pre-process both the 'train' and 'valid' datasets.
        """

        processed_splits = []

        if len(splits) > 3 or len(splits) < 1:
            raise ValueError('Invalid number of splits passed!')

        for split in splits:
            if split == 'train':
                data = self.train_data
            elif split == 'valid':
                data = self.val_data
            elif split == 'test':
                data = self.test_data
            else:
                raise ValueError('Invalid split passed. Refer to func. documentation.')
            
            processed_split = self._preprocess_split(data)
            processed_splits.append(processed_split)

        if len(splits) == 1:
            return processed_splits[0]
        elif len(splits) == 2:
            return tuple(processed_splits)
        else:
            return tuple(processed_splits)
        

    def tokenize(self, tokenizer_type: str, splits = ['train'], tokenizer=None) -> Union[List[str], 
                                                                                         Tuple[List[str], List[str]], 
                                                                                         Tuple[List[str], List[str], List[str]]
                                                                                         ]:
        """
        Tokenizes the dataset based on the specified tokenizer type and the given splits (e.g., train, valid, test).

        Parameters:
        - tokenizer_type (str): The type of tokenization to apply. Choices are:
            - 'whitespace': Tokenizes the text based on whitespace.
            - 'characters': Tokenizes the text into individual characters.
            - 'nltk': Tokenizes the text using NLTK's word_tokenize method.
            - 'pretrained': Tokenizes the text using a pretrained tokenizer from HuggingFace's Transformers library.
        - splits (list): A list of dataset splits to tokenize. Can contain any of the following:
            - 'train': Tokenize the training set.
            - 'valid': Tokenize the validation set.
            - 'test': Tokenize the test set.
            Default is ['train'].
        - tokenizer (str or None): If using a pretrained tokenizer (i.e., 'pretrained' tokenizer_type), provide the model name (e.g., 'bert-base-uncased').
            Otherwise, set to None. This argument is ignored for other tokenization types.

        Returns:
        - List[str]: A list of tokenized text for the selected dataset split(s).
            If multiple are provided, it returns a tuple of tokenized lists.
        
        Example:
        >>> tokenizer = Tokenizer()
        >>> tokenized_data = tokenizer.tokenize('whitespace', splits=['train', 'valid'])
        >>> print(tokenized_data)

        This function will tokenize the 'train' and 'valid' datasets using whitespace tokenization.
        """


        if len(splits) > 3 or len(splits) < 1:
            raise ValueError('Invalid number of splits passed!')

        tokenized_splits = []

        tokenizer_instance = TokenizerFactory.get_tokenizer(tokenizer_type, pretrained_model=tokenizer)

        for split in splits:
            if split == 'train':
                data = self.train_data
            elif split == 'valid':
                data = self.val_data
            elif split == 'test':
                data = self.test_data
            else:
                raise ValueError('Invalid split passed. Refer to func. documentation.')

            text_data = data['TEXT']

            tokenized_data = [tokenizer_instance.tokenize(text) for text in text_data]
            tokenized_splits.append(tokenized_data)

        if len(splits) == 1:
            return tokenized_splits[0]
        elif len(splits) == 2:
            return tuple(tokenized_splits)
        else:
            return tuple(tokenized_splits)


