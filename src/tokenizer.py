import string
from typing import List


class Tokenizer:
    def __init__(self) -> None:
        """
        Initialize a Tokenizer that maps characters to unique ids."""
        self.vocab = list(string.printable)
        self.token_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_token = {idx: char for idx, char in enumerate(self.vocab)}

    def tokenize(self, text: str) -> List[int]:
        """
        Convert a text into a list of integers.
        """
        return [self.token_to_id[char] for char in text if char in self.token_to_id]

    def detokenize(self, tokens: List[int]) -> str:
        """
        Convert a list of integers back into a string.
        """
        return "".join(
            self.id_to_token[token] for token in tokens if token in self.id_to_token
        )

    def vocab_size(self) -> int:
        """
        Return the vocabulary size.
        """
        return len(self.vocab)
