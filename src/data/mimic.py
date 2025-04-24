import os
from pathlib import Path
import random
import re
from typing import Dict, List, Tuple, Union
import nltk

import pandas as pd
from tqdm import tqdm

from ..tokenizer.factory import TokenizerFactory
from .base import BaseDataset
from env import ProjectPaths
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import json

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

class MIMIC_IV(BaseDataset):
    def __init__(self, name):
        super().__init__(name)
        print(f'MIMIC-IV dataset initialized with name: {self.name}')

    
    def compute_class_weights(self):
        """
        Computes class weights using sklearn's compute_class_weight and aligns with class index.
        """

        # Flatten all ICD codes
        all_codes = [code for code in self.data['icd_code']]

        # Class labels and compute weights
        class_labels = list(self.class_to_idx.keys())  # list of ICD codes
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array(class_labels),
            y=np.array(all_codes)
        )

        # Map ICD -> weight
        code_to_weight = dict(zip(class_labels, weights))

        # Reorder weights by class index (0 to num_classes-1)
        num_classes = len(self.class_to_idx)
        class_weights_ordered = [0.0] * num_classes
        for code, idx in self.class_to_idx.items():
            class_weights_ordered[idx] = code_to_weight[code]

        # Save to JSON
        output_path = ProjectPaths.DATASET_DIR.value / 'MIMIC-IV' / 'class_weights.json'
        with open(output_path, 'w') as f:
            json.dump(class_weights_ordered, f)

        print(f"Saved sklearn-computed class weights for {len(class_weights_ordered)} classes to {output_path}")



    def load_dataset(self) -> Tuple[pd.DataFrame]:
        """
        Loads the MIMIC-IV dataset from local.
        """
        mimic_dir = ProjectPaths.DATASET_DIR.value / 'MIMIC-IV' / 'raw_dataset'
        required_files = {
            'diagnosis.csv',
            'discharge.csv',
            'edstays.csv'
        }

        if (mimic_dir / 'mimic_merged.csv').exists():
            self.data = pd.read_csv(mimic_dir / 'mimic_merged.csv')
            return self.data

        missing_files = [file for file in required_files if not (mimic_dir / file).exists()]

        if missing_files:
            raise FileNotFoundError(
                f"Missing the following required files in {mimic_dir}:\n"
                f"{', '.join(missing_files)}\n\n"
                "Please download the MIMIC-IV-ED and MIMIC-IV-NOTE dataset from PhysioNet:\n"
                "https://physionet.org/content/mimic-iv-ed/2.2/ed/#files-panel \n"
                "https://www.physionet.org/content/mimic-iv-note/2.2/ \n"
                "and place the required files in the above directory."
            )

        discharge_df = pd.read_csv(mimic_dir / 'discharge.csv')
        diagnosis_df = pd.read_csv(mimic_dir / 'diagnosis.csv')
        edstays_df = pd.read_csv(mimic_dir / 'edstays.csv')

        self.data = self.combine_into_one_df(discharge_df, diagnosis_df, edstays_df)
        self.data.to_csv(mimic_dir / 'mimic_merged.csv', index=False)
        return self.data
    
    @staticmethod
    def custom_sentence_split(text: str):
        # Split on periods, newlines, or colons followed by a space or digit/uppercase
        return [s.strip() for s in re.split(r'[\n\.]+(?=\s*\d|[A-Z])', text) if s.strip()]

    @staticmethod
    def select_top_sentences_tfidf(note_text: str, max_words: int = 50) -> str:
        """
        Uses TF-IDF to score sentences in note_text and selects the top sentences whose total word count is <= max_words.
        """
        max_words += max_words

        if not isinstance(note_text, str) or not note_text.strip():
            return ""

        sentences = MIMIC_IV.custom_sentence_split(note_text)

        if not sentences:
            return ""
        
        if len(sentences) == 1:
            words = sentences[0].split()
            return " ".join(words[:max_words])

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(sentences)

        # Score each sentence by summing its TF-IDF weights
        scores = X.sum(axis=1).A1

        # Sort sentences in descending order by score
        top_sentences = [sent for _, sent in sorted(zip(scores, sentences), reverse=True)]
        selected = []
        count = 0

        for sent in top_sentences:
            words = sent.split()
            n = len(words)

            if count + n <= max_words:
                selected.append(" ".join(words))
                count += n
            else:
                remaining_space = max_words - count
                if remaining_space > 0:
                    selected.append(" ".join(words[:remaining_space]))
                    count += remaining_space
                break

        return selected    


    def combine_into_one_df(self,
                              discharge_df: pd.DataFrame,
                              diagnosis_df: pd.DataFrame,
                              edstays_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combines discharge notes, diagnosis, and ED stay data from MIMIC-IV into one DataFrame
        using precise filtering based on subject_id, hadm_id, and stay_id.
        """
        records = []

        for _, row in discharge_df.iterrows():
            subject_id = row['subject_id']
            hadm_id = row['hadm_id']
            text = row['text']

            # 1. Find the stay_id from edstays_df where subject_id and hadm_id match
            stay_row = edstays_df[
                (edstays_df['subject_id'] == subject_id) &
                (edstays_df['hadm_id'] == hadm_id)
            ]
            if stay_row.empty:
                continue  # skip if no matching stay_id

            stay_id = stay_row.iloc[0]['stay_id']

            # 2. Find diagnosis info from diagnosis_df using subject_id and stay_id
            diagnosis_rows = diagnosis_df[
                (diagnosis_df['subject_id'] == subject_id) &
                (diagnosis_df['stay_id'] == stay_id)
            ]
            if diagnosis_rows.empty:
                continue  # skip if no matching diagnosis

            for _, diag_row in diagnosis_rows.iterrows():
                records.append({
                    'subject_id': subject_id,
                    'hadm_id': hadm_id,
                    'stay_id': stay_id,
                    'text': text,
                    'icd_code': diag_row['icd_code'],
                    'icd_title': diag_row['icd_title']
                })
        combined_df = pd.DataFrame(records, index=None)
        return combined_df

    def split_dataset(self, train_size=0.8, val_size=0.1):
        """Randomly splits self.data into train, val, and test sets."""
        data_list = self.data.to_dict(orient='records')
        random.shuffle(data_list)
        total_len = len(data_list)
        train_end = int(total_len * train_size)
        val_end = int(total_len * (train_size + val_size))

        self.train_data = pd.DataFrame(data_list[:train_end])
        self.val_data = pd.DataFrame(data_list[train_end:val_end])
        self.test_data = pd.DataFrame(data_list[val_end:])
        return self.train_data, self.val_data, self.test_data


    def group_multilabel_data(self):
        """
        Groups the dataset by subject_id, hadm_id, stay_id, and text to form multi-label entries.
        ICD codes for each group are aggregated into a list.
        """
        # Group by identifiers and text
        grouped = self.data.groupby(
            ['subject_id', 'hadm_id', 'stay_id', 'text'],
            as_index=False
        ).agg({
            'icd_code': lambda x: list(set(x)),  # Remove duplicates within a note
            'icd_title': lambda x: list(set(x))
        })

        # Update internal data
        self.data = grouped


    def convert_class_to_idx(self):
        """
        Groups ICD codes into generalized classes.
        """
        from collections import defaultdict

        def group_icd(icd_code: str) -> str:
            icd_code = icd_code.strip().upper()
            if icd_code.isdigit():
                return icd_code[:3] if len(icd_code) > 3 else icd_code
            else:
                return icd_code[:4] if len(icd_code) > 4 else icd_code

        # Backup original ICD code
        self.data['original_icd_code'] = self.data['icd_code']
        self.data['icd_code'] = self.data['icd_code'].apply(group_icd)

        # Build maps for ICD classes
        self.icd_to_class = {
            row['original_icd_code']: row['icd_code']
            for _, row in self.data[['original_icd_code', 'icd_code']].drop_duplicates().iterrows()
        }
        icd_to_classes_map = defaultdict(set)
        for _, row in self.data.iterrows():
            icd_to_classes_map[row['icd_code']].add(row['icd_title'])

        self.icd_to_classes = {k: sorted(list(v)) for k, v in icd_to_classes_map.items()}
        unique_classes = sorted(self.icd_to_classes.keys())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        # self.class_to_idx['unk'] = len(unique_classes)

        self.compute_class_weights()

        return self.class_to_idx, self.icd_to_class, self.icd_to_classes


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        if idx >= len(self.data) or idx < 0:
            raise ValueError('Index out of bounds')
        return self.data.iloc[idx]


    @staticmethod
    def extract_preprocessed_hpi_static(text: str, summary_len: int) -> str:
        # Only work with valid strings
        if not isinstance(text, str):
            return ""

        # Lowercase and extract possible HPI section
        text = text.lower()
        match = re.search(r'(?:history of present illness|hpi)(.*?)(?:\n\n|\Z)', text, re.DOTALL)
        hpi_section = match.group(1).strip() if match else text

        # Remove non-alphanumeric characters (leave only a-z, 0-9, and whitespace)
        hpi_section = re.sub(r'[^a-z0-9\s]', '', hpi_section)

        # Tokenize by whitespace
        tokens = hpi_section.split()

        # Return truncated text
        return ' '.join(tokens[:summary_len])




    def _preprocess_split(self, data: pd.DataFrame, summary_len: int) -> pd.DataFrame:
        # Drop rows with non-string 'text' entries
        data = data[data['text'].apply(lambda x: isinstance(x, str))]

        # Define row-wise preprocessing logic
        def preprocess_row(row):
            text = row['text']
            processed = MIMIC_IV.extract_preprocessed_hpi_static(text, summary_len)
            processed = BaseDataset.remove_stop_words(processed)
            processed = BaseDataset.lemmatizer(processed)
            row['text'] = processed
            return row

        # Apply preprocessing in parallel
        return data.parallel_apply(preprocess_row, axis=1)


    def preprocess(self, splits=['train'], summary_len: int = 50) -> Union[pd.DataFrame, Tuple[pd.DataFrame, ...]]:
        """
        Preprocesses the dataset splits.
        """
        processed_splits = []
        if len(splits) > 3 or len(splits) < 1:
            raise ValueError('Invalid number of splits passed!')
        for split in splits:
            if split == 'train':
                data = self.train_data
                processed_split = self._preprocess_split(data, summary_len)
                self.train_data = processed_split
            elif split == 'valid':
                data = self.val_data
                processed_split = self._preprocess_split(data, summary_len)
                self.val_data = processed_split
            elif split == 'test':
                data = self.test_data
                processed_split = self._preprocess_split(data, summary_len)
                self.test_data = processed_split
            else:
                raise ValueError('Invalid split passed. Refer to func. documentation.')
            processed_splits.append(processed_split)
        return processed_splits[0] if len(splits) == 1 else tuple(processed_splits)

    
    def tokenize(self, tokenizer_type: str, splits=['train'], **kwargs) -> Union[List[str], Tuple[List[str], ...]]:
        """
        Tokenizes the dataset based on the specified tokenizer type for given splits.
        """
        if len(splits) > 3 or len(splits) < 1:
            raise ValueError('Invalid number of splits passed!')
        tokenized_splits = []
        for split in splits:
            if split == 'train':
                data = self.train_data
            elif split == 'valid':
                data = self.val_data
            elif split == 'test':
                data = self.test_data
            else:
                raise ValueError('Invalid split passed. Refer to func. documentation.')
            text_data = data['text']
            tokenizer_instance = TokenizerFactory.get_tokenizer(tokenizer_type, **kwargs)
            if tokenizer_type == 'pretrained':
                return tokenizer_instance.tokenize()
            else:
                tokenized_data = text_data.parallel_apply(lambda text: tokenizer_instance.tokenize(text))
                tokenized_splits.append(tokenized_data)
        return tokenized_splits[0] if len(splits) == 1 else tuple(tokenized_splits)

    def embed(self):
        """Embed using TF-IDF on context column"""
        pass

    def save_embeddings(self, path: str):
        """Save embeddings as .npy and the vectorizer as a pickle file."""
        print(f"Embeddings saved to {path}.npy and vectorizer to {path}_vectorizer.pkl")
