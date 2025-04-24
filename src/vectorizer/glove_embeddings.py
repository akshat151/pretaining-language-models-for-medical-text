import torch
import numpy as np
from typing import List, Union, Optional
from .base import Embedding

class GloVeEmbedding(Embedding):
    """Embedding model that uses GloVe pre-trained embeddings."""
    
    def __init__(self, model_path: str, embedding_dim: int = 100, 
                 vocab_size_limit=None, context_window = 7, external_vocab: Optional[dict] = None):
        """
        Initializes the GloVe embedding model.
        
        Args:
            model_path (str): Path to the GloVe embeddings file.
            embedding_dim (int): The dimensionality of the embeddings (default 100).
            vocab_size_limit (int, optional): Maximum vocabulary size (default is None, no limit).
            external_vocab (dict, optional): An external mapping from word to index to enforce consistency
                                             with your training tokenization process.
        """
        self.embedding_dim = embedding_dim
        self.vocab_size_limit = vocab_size_limit
        self.external_vocab = external_vocab
        
        # Load GloVe embeddings
        self.embeddings_index = self.load_glove_embeddings(model_path)
        
        # Build vocabulary and embedding matrix based on external vocabulary if provided.
        if self.external_vocab is not None:
            (self.word_to_idx, self.idx_to_word, 
             self.embedding_matrix) = self.build_vocab_from_external_vocab(self.external_vocab)
        else:
            self.word_to_idx, self.idx_to_word, self.embedding_matrix = self.build_vocab_from_embeddings()

        # Convert the embedding matrix into a torch tensor
        self.embedding_matrix = torch.tensor(self.embedding_matrix, dtype=torch.float32)
        self.context_window = context_window

    def load_glove_embeddings(self, glove_file_path):
        """Load GloVe embeddings from a file."""
        embeddings_index = {}
        with open(glove_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        return embeddings_index

    def build_vocab_from_embeddings(self):
        """Build vocabulary and embedding matrix from the loaded GloVe embeddings."""
        word_to_idx = {"<PAD>": 0, "<UNK>": 1}
        idx_to_word = {0: "<PAD>", 1: "<UNK>"}
        embedding_matrix = []

        # Initialize with <PAD> and <UNK> embeddings
        embedding_matrix.append(np.zeros(self.embedding_dim))  # <PAD> embedding (zeros)
        embedding_matrix.append(np.random.randn(self.embedding_dim))  # Random <UNK> embedding

        # Ensure the <UNK> and <PAD> tokens exist in embeddings_index
        self.embeddings_index['<UNK>'] = np.random.randn(self.embedding_dim)
        self.embeddings_index['<PAD>'] = np.zeros(self.embedding_dim)

        for word, embedding in self.embeddings_index.items():
            if self.vocab_size_limit and len(word_to_idx) >= self.vocab_size_limit:
                break
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
                idx_to_word[len(idx_to_word)] = word
                embedding_matrix.append(embedding)

        embedding_matrix = np.array(embedding_matrix)
        return word_to_idx, idx_to_word, embedding_matrix

    def build_vocab_from_external_vocab(self, external_vocab: dict):
        """
        Build vocabulary and embedding matrix using an external vocabulary.
        The external_vocab should be a mapping from token to index. The resulting embedding matrix
        will be built in that order.
        """
        word_to_idx = {}
        idx_to_word = {}
        embedding_matrix = []
        
        # Reserve indices 0 and 1 for <PAD> and <UNK>
        word_to_idx["<PAD>"] = 0
        idx_to_word[0] = "<PAD>"
        embedding_matrix.append(np.zeros(self.embedding_dim))
        
        word_to_idx["<UNK>"] = 1
        idx_to_word[1] = "<UNK>"
        embedding_matrix.append(self.embeddings_index.get("<UNK>", np.random.randn(self.embedding_dim)))
        
        # Sort external_vocab by their index to maintain consistency
        sorted_words = sorted(external_vocab, key=lambda token: external_vocab[token])
        for word in sorted_words:
            if word in ["<PAD>", "<UNK>"]:
                continue
            cur_idx = len(word_to_idx)
            word_to_idx[word] = cur_idx
            idx_to_word[cur_idx] = word
            # Use the glove embedding if available, otherwise use a random vector
            if word in self.embeddings_index:
                vector = self.embeddings_index[word]
            else:
                vector = np.random.randn(self.embedding_dim)
            embedding_matrix.append(vector)
            
            if self.vocab_size_limit and len(word_to_idx) >= self.vocab_size_limit:
                break

        embedding_matrix = np.array(embedding_matrix)
        return word_to_idx, idx_to_word, embedding_matrix

    def embed(self, tokenized_data: Union[str, List[str]]) -> np.ndarray:
        """
        Embed a single text or a list of texts using GloVe embeddings.
        
        Args:
            tokenized_data (str or list of str): Tokenized text(s).
        
        Returns:
            np.ndarray: A 2D array of shape (seq_len, embedding_dim) representing the embeddings for the input.
        """
        if isinstance(tokenized_data, str):
            tokenized_data = tokenized_data.split()  # Convert string to list of tokens
        
        embeddings = []
        for token in tokenized_data:
            embedding = self.embeddings_index.get(token, self.embeddings_index['<UNK>'])
            embeddings.append(embedding)
        return np.array(embeddings)
    
    
    def token_indices(self, tokenized_corpus: List[str], abbreviation: str = None):
        token_indices = []
        start = -1
        end = -1
        for idx, token in enumerate(tokenized_corpus):
            if token == abbreviation:
                if start == -1:
                    start = idx
                elif end == -1:
                    end = idx

            token_indices.append(
                self.word_to_idx.get(token, self.word_to_idx['<UNK>'])  # If not found return idx of <UNK>
            )

        if abbreviation:
            start = max(start - self.context_window, 0)
            end = min(end + self.context_window, len(tokenized_corpus))
        else:
            start = 0
            end = len(tokenized_corpus)

        return token_indices[start: end]