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
    assert len(tokens) == len(text)
    assert all(isinstance(token, int) for token in tokens)
    assert reconstructed == text


def test_empty_text(tokenizer: Tokenizer, empty_text: str) -> None:
    """Test the Tokenizer with an empty string."""
    tokens = tokenizer.tokenize(empty_text)
    reconstructed = tokenizer.detokenize(tokens)
    assert len(tokens) == 0
    assert reconstructed == empty_text


def test_unknown_char(tokenizer: Tokenizer, unknown_char: str) -> None:
    """Test the Tokenizer with an unknown character."""
    tokens = tokenizer.tokenize(unknown_char)
    reconstructed = tokenizer.detokenize(tokens)
    assert len(tokens) == 0
    assert reconstructed == ""
