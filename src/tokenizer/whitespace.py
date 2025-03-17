

from typing import List
from tokenizer.base import Tokenizer

class WhitespaceTokenizer(Tokenizer):
    """
    Tokenizer that splits text by whitespace.
    """
    def tokenize(self, text: str) -> List[str]:
        return text.split()
