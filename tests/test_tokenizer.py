import pytest

from src.tokenizer import Tokenizer


@pytest.fixture
def text() -> str:
    return "Hello, World! 123"


@pytest.fixture
def empty_text() -> str:
    return ""


@pytest.fixture
def unknown_char() -> str:
    return "👍"


@pytest.fixture
def tokenizer() -> Tokenizer:
    return Tokenizer()


def test_tokenizer(tokenizer: Tokenizer, text: str) -> None:
    """Test the basic functionality of the Tokenizer."""
    tokens = tokenizer.tokenize(text)
    reconstructed = tokenizer.detokenize(tokens)
    assert len(reconstructed) == len(text)
    assert all(isinstance(token, int) for token in tokens)
    assert reconstructed == text


def test_empty_text(tokenizer: Tokenizer, empty_text: str) -> None:
    """Test the Tokenizer with an empty string."""
    tokens = tokenizer.tokenize(empty_text)
    reconstructed = tokenizer.detokenize(tokens)
    assert len(tokens) == 2
    assert len(reconstructed) == 0
    assert reconstructed == empty_text


def test_unknown_char(tokenizer: Tokenizer, unknown_char: str) -> None:
    """Test the Tokenizer with an unknown character."""
    tokens = tokenizer.tokenize(unknown_char)
    reconstructed = tokenizer.detokenize(tokens)
    assert len(tokens) == 2
    assert len(reconstructed) == 0
    assert reconstructed == ""


def test_seq_length(tokenizer: Tokenizer, text: str, empty_text: str) -> None:
    """Test the Tokenizer with a sequence length."""
    seq_length = 10
    tokens = tokenizer.tokenize(text, seq_length=seq_length)
    assert len(tokens) == seq_length
    assert all(isinstance(token, int) for token in tokens)
    assert tokens[0] == tokenizer.get_sos_token_id()
    assert tokens[-1] == tokenizer.get_eos_token_id()
    assert tokenizer.tokenize(empty_text, seq_length=seq_length) == [
        tokenizer.get_sos_token_id()
    ] + [tokenizer.get_pad_token_id()] * (seq_length - 2) + [
        tokenizer.get_eos_token_id()
    ]
