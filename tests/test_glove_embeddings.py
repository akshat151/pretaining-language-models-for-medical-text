import os
import tempfile
import numpy as np
import pytest
from src.vectorizer.glove_embeddings import GloVeEmbedding


@pytest.fixture
def dummy_glove_file():
    glove_content = """\
the 0.1 0.2 0.3
cat 0.4 0.5 0.6
sat 0.7 0.8 0.9
on 1.0 1.1 1.2
mat 1.3 1.4 1.5
"""
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write(glove_content)
        f.flush()
        yield f.name
    os.remove(f.name)


def test_glove_loading(dummy_glove_file):
    emb = GloVeEmbedding(model_path=dummy_glove_file, embedding_dim=3)
    assert 'cat' in emb.embeddings_index
    assert emb.embeddings_index['the'].tolist() == [0.1, 0.2, 0.3]


def test_vocab_building(dummy_glove_file):
    emb = GloVeEmbedding(model_path=dummy_glove_file, embedding_dim=3)
    assert emb.word_to_idx["<PAD>"] == 0
    assert emb.word_to_idx["<UNK>"] == 1
    assert "cat" in emb.word_to_idx
    assert emb.embedding_matrix.shape[1] == 3


def test_embedding_lookup(dummy_glove_file):
    emb = GloVeEmbedding(model_path=dummy_glove_file, embedding_dim=3)
    sequence = ["the", "cat", "flew"]
    embedded = emb.embed(sequence)
    assert embedded.shape == (3, 3)
    np.testing.assert_array_almost_equal(embedded[0], [0.1, 0.2, 0.3])
    # "flew" not in GloVe, should get <UNK>
    unk_vector = emb.embeddings_index["<UNK>"]
    np.testing.assert_array_almost_equal(embedded[2], unk_vector)


def test_build_vocab_from_external(dummy_glove_file):
    external_vocab = {"the": 2, "cat": 3, "on": 4, "newword": 5}
    emb = GloVeEmbedding(
        model_path=dummy_glove_file,
        embedding_dim=3,
        external_vocab=external_vocab
    )

    # Must include <PAD> and <UNK>
    assert emb.word_to_idx["<PAD>"] == 0
    assert emb.word_to_idx["<UNK>"] == 1
    assert emb.word_to_idx["the"] == 2
    assert emb.word_to_idx["newword"] == 5
    assert emb.embedding_matrix.shape[0] == 6  # PAD + UNK + 4 words


def test_token_indices_without_abbreviation(dummy_glove_file):
    emb = GloVeEmbedding(model_path=dummy_glove_file, embedding_dim=3)
    tokens = ["the", "cat", "sat"]
    indices = emb.token_indices(tokens)
    assert isinstance(indices, list)
    assert len(indices) == len(tokens)
    assert all(isinstance(i, int) for i in indices)


def test_token_indices_with_abbreviation_windowing(dummy_glove_file):
    emb = GloVeEmbedding(model_path=dummy_glove_file, embedding_dim=3, context_window=1)
    tokens = ["the", "cat", "sat", "on", "the", "mat"]
    indices = emb.token_indices(tokens, abbreviation="sat")

    # Expecting one token before and one after "sat"
    expected_window = tokens[1:4]  # ["cat", "sat", "on"]
    expected_indices = [emb.word_to_idx.get(t, emb.word_to_idx["<UNK>"]) for t in expected_window]
    assert indices == expected_indices
