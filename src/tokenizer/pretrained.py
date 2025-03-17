from typing import List
from tokenizer.base import Tokenizer
from transformers import BertTokenizer

class PretrainedTokenizer(Tokenizer):
    """
    Tokenizer that uses a pretrained model from HuggingFace's Transformers library.
    """
    def __init__(self, pretrained_model: str):
        self.tokenizer: PretrainedTokenizer = BertTokenizer.from_pretrained(pretrained_model)
    
    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)