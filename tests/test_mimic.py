import pytest
from src.data.mimic import MIMIC_IV

def test_custom_sentence_split():
    text = "History of present illness. Patient admitted.\nNext line sentence."
    sentences = MIMIC_IV.custom_sentence_split(text)
    # Should split by period or newline
    assert isinstance(sentences, list)
    assert len(sentences) >= 2

def test_select_top_sentences_tfidf():
    note = "This is a test sentence. Another important sentence. And a less important one."
    # Use a small max_words value.
    top = MIMIC_IV.select_top_sentences_tfidf(note, max_words=10)
    # It returns a list of sentences.
    assert isinstance(top, list)
    # The total words in selected sentences should not exceed max_words
    total_words = sum(len(s.split()) for s in top)
    assert total_words <= 10