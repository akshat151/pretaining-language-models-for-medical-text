from enum import Enum
import os
from pathlib import Path
import gensim
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Optional
from gensim.models import Word2Vec, FastText
from .base import Embedding
from scipy.sparse import save_npz, load_npz

class EmbeddingAlgorithm(Enum):
    WORD2VEC = "word2vec"
    TF_IDF = "tfidf"
    FASTTEXT = "fasttext"

class TrainableEmbedding(Embedding):
    """
    A configurable embedding model that can be trained on a corpus using different algorithms.
    
    Supported algorithms:
    - Word2Vec
    - FastText
    - TF-IDF
    
    Usage:
        embedding_model = TrainableEmbedding(
            tokenized_corpus=texts,
            algorithm="fasttext", 
            vector_size=100, 
            window=5,
            min_count=2
        )
        embedding_model.train()
        embeddings = embedding_model.embed(text)
    """
    
    def __init__(self,
                 tokenized_corpus: List[List[str]],  # Each document is a list of tokens
                 algorithm: str,
                 vector_size: int = 100,
                 window: int = 5,
                 min_count: int = 2,
                 special_tokens: Optional[List[str]] = None):
        """
        Initialize the trainable embedding model.
        
        Args:
            tokenized_corpus: List of documents, where each document is a list of tokens.
            algorithm: The embedding algorithm to use ('word2vec', 'glove', 'fasttext', 'tfidf')
            vector_size: The dimensionality of the embeddings
            window: The window size for Word2Vec, GloVe, or FastText
            min_count: Minimum number of occurrences for a word to be considered
            special_tokens: List of special tokens for TF-IDF, if necessary
        """
        try:
            self.algorithm = algorithm.lower()
        except ValueError:
            raise ValueError(f"Algorithm must be one of: {[algo.value for algo in EmbeddingAlgorithm]}")

        # Create model directory
        self.model_dir = Path('trained_models/embeddings/trained')
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.special_tokens = special_tokens or ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        self.aggregation_method = 'average'
        
        if isinstance(tokenized_corpus, list):
            self.tokenized_corpus = tokenized_corpus
        else:
            raise TypeError("corpus must be a list of tokenized documents")

        self.model = None
        self.vectorizer = None  # For TF-IDF
        self.trained = False

    def is_trained(self) -> bool:
        """Check if the model has been trained and saved"""
        if self.algorithm == EmbeddingAlgorithm.TF_IDF:
            path = f'trained_models/embeddings/trained/trained_{self.algorithm}.npz'
        else:
            path = f'trained_models/embeddings/trained/trained_{self.algorithm}.model'

        if os.path.exists(path):
            return True
        
        return False
        
    def train(self) -> None:
        """Train the embedding model on the corpus"""
        if self.algorithm == EmbeddingAlgorithm.TF_IDF.value:
            self.train_tfidf()
        elif self.algorithm == EmbeddingAlgorithm.WORD2VEC.value:
            self.train_word2vec()
        elif self.algorithm == EmbeddingAlgorithm.FASTTEXT.value:
            self.train_fasttext()
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        self.trained = True
        self.save()

        
    def train_word2vec(self) -> None:
        """Train Word2Vec embeddings on the corpus"""
        self.model = Word2Vec(
            self.tokenized_corpus, 
            vector_size=self.vector_size, window=self.window, min_count=self.min_count)
    
    def train_fasttext(self) -> None:
        """Train FastText embeddings on the corpus"""
        self.model = FastText(
            self.tokenized_corpus, 
            vector_size=self.vector_size, 
            window=self.window, 
            min_count=self.min_count
            )

    def train_tfidf(self) -> None:
        """Train TF-IDF vectorizer on the corpus"""
        documents = [' '.join(tokens) for tokens in self.tokenized_corpus]
        self.vectorizer = TfidfVectorizer()
        self.model = self.vectorizer.fit_transform(documents)
        
    def save(self) -> None:
        """Save the trained model to disk"""
        if self.algorithm == EmbeddingAlgorithm.TF_IDF.value:
            path = self.model_dir / f'trained_{self.algorithm}.npz'
            save_npz(str(path), self.model)
            # Also save the vectorizer
            import pickle
            with open(self.model_dir / f'trained_{self.algorithm}_vectorizer.pkl', 'wb') as f:
                pickle.dump(self.vectorizer, f)
        else:
            path = self.model_dir / f'trained_{self.algorithm}.model'
            self.model.save(str(path))

    def load(self) -> None:
        """Load a trained model from disk"""
        if self.algorithm == EmbeddingAlgorithm.TF_IDF.value:
            path = self.model_dir / f'trained_{self.algorithm}.npz'
            self.model = load_npz(str(path))
            # Load the vectorizer
            import pickle
            with open(self.model_dir / f'trained_{self.algorithm}_vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
        else:
            path = self.model_dir / f'trained_{self.algorithm}.model'
            self.model = (gensim.models.FastText.load(str(path)) 
                         if self.algorithm == EmbeddingAlgorithm.FASTTEXT.value 
                         else gensim.models.Word2Vec.load(str(path)))


    def embed(self, tokenized_corpus: List[List[str]]) -> np.ndarray:
        """
        Generate embeddings for a list of documents in the tokenized corpus.

        Args:
            tokenized_corpus: List of documents where each document is a list of tokens.

        Returns:
            Embeddings of all documents, each represented by a vector.
        """
        if not self.trained:
            if not any(self.model_dir.glob(f'trained_{self.algorithm}.*')):
                self.train()
            else:
                self.load()
                self.trained = True

        if self.algorithm == EmbeddingAlgorithm.TF_IDF.value:
            documents = [' '.join(tokens) for tokens in tokenized_corpus]
            embeddings = self.vectorizer.transform(documents)
            return embeddings.toarray()
        else:
            all_embeddings = []
            for doc in tokenized_corpus:
                doc_embeddings = []
                for token in doc:
                    if token in self.model.wv:
                        doc_embeddings.append(self.model.wv[token])
                    else:
                        doc_embeddings.append(np.zeros(self.vector_size))
                        
                if doc_embeddings:
                    doc_embedding = np.mean(doc_embeddings, axis=0)
                else:
                    doc_embedding = np.zeros(self.vector_size)
                all_embeddings.append(doc_embedding)
            
            return np.array(all_embeddings)