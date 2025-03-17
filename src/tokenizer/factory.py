from tokenizer.base import Tokenizer
from tokenizer.character import CharacterTokenizer
from tokenizer.nltk import NLTKTokenizer
from tokenizer.pretrained import PretrainedTokenizer
from tokenizer.whitespace import WhitespaceTokenizer


class TokenizerFactory:
    """
    Factory class that maps tokenizer types to specific Tokenizer classes.
    """
    tokenizer_map = {
        'whitespace': WhitespaceTokenizer,
        'characters': CharacterTokenizer,
        'nltk': NLTKTokenizer,
        'pretrained': PretrainedTokenizer,
    }

    @staticmethod
    def get_tokenizer(tokenizer_type: str, pretrained_model: str = None) -> Tokenizer:
        """
        Returns an instance of the appropriate tokenizer based on the tokenizer type.

        Parameters:
        - tokenizer_type (str): Type of tokenizer ('whitespace', 'characters', 'nltk', 'pretrained', etc).
        - pretrained_model (str): The name of the pretrained tokenizer (e.g., 'bert-base-uncased') if using pretrained tokenization.

        Returns:
        - Tokenizer: The corresponding tokenizer object.
        """
        if tokenizer_type == 'pretrained' and pretrained_model is None:
            raise ValueError("Please provide a pretrained model name when using pretrained tokenization.")
        
        if tokenizer_type in TokenizerFactory.tokenizer_map:
            # If pretrained tokenizer, pass the model name to the PretrainedTokenizer
            if tokenizer_type == 'pretrained':
                return TokenizerFactory.tokenizer_map[tokenizer_type](pretrained_model)
            
            return TokenizerFactory.tokenizer_map[tokenizer_type]()
        
        raise ValueError(f"Unknown tokenizer type '{tokenizer_type}'\
                        Check `tokenizer_map` for tokenizers available")