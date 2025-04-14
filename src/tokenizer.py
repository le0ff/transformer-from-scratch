import string
from typing import List


class Tokenizer:
    def __init__(self) -> None:
        """
        Initialize a Tokenizer that maps characters to unique ids."""
        # Special tokens
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        self.special_tokens = [self.pad_token, self.sos_token, self.eos_token]
        self.vocab = self.special_tokens + list(string.printable)
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.vocab)}

    def tokenize(self, text: str, max_length: int = None) -> List[int]:
        """
        Convert a text into a list of integers.
        """
        tokens = (
            [self.char_to_id[self.sos_token]]
            + [self.char_to_id[char] for char in text if char in self.char_to_id]
            + [self.char_to_id[self.eos_token]]
        )
        if max_length:
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens += [self.char_to_id[self.pad_token]] * (max_length - len(tokens))

        return tokens

    def detokenize(self, char_ids: List[int]) -> str:
        """
        Convert a list of integers back into a string.
        """
        chars = []
        for idx in char_ids:
            char = self.id_to_char.get(idx, "")
            if char in self.special_tokens:
                continue
            chars.append(char)
        return "".join(chars)

    def vocab_size(self) -> int:
        """
        Return the vocabulary size.
        """
        return len(self.vocab)

    def get_pad_token_id(self) -> int:
        """
        Return the id of the padding token.
        """
        return self.char_to_id[self.pad_token]

    def get_sos_token_id(self) -> int:
        """
        Return the id of the start of sequence token.
        """
        return self.char_to_id[self.sos_token]

    def get_eos_token_id(self) -> int:
        """
        Return the id of the end of sequence token.
        """
        return self.char_to_id[self.eos_token]
