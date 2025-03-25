from typing import List, Union
import torch
from transformers import AutoModel, AutoTokenizer
from .base import Embedding
import os
from gensim.models import FastText
import numpy as np

class BioWordVecModel(Embedding):
    """BERT-based embedding model."""
    def __init__(self, model_path: str):
        self.path = model_path
        self.model = self.load_model(model_path)
        self.aggregation_method = 'average' # TODO: Take this value from config

    def load_model(self, model_path):
        if os.path.exists(model_path):
            self.model = FastText.load_fasttext_format(model_path)
        else:
            raise ValueError(f'Unable to find BioWordVec model in path: {model_path}')

    def embed(self, tokenized_data):
        all_embeddings = []
        
        for doc in tokenized_data:
            doc_embeddings = []
            for token in doc:
                if token in self.model:
                    doc_embeddings.append(self.model[token])
                else:
                    # Use subword information (available in FastText-based BioWordVec) to handle OOV tokens
                    # doc_embeddings.append(np.zeros(self.model.vector_size))
                    # doc_embeddings.append(self.model.get_word_vector('[UNK]'))
                    doc_embeddings.append(self.model.get_word_vector(token))

            doc_embedding = Embedding.build_doc_embedding(doc_embeddings, self.aggregation_method)
            all_embeddings.append(doc_embedding)

        all_embeddings = np.array(all_embeddings)

        return all_embeddings


