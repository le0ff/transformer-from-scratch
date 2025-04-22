import numpy as np
import pytest
from numpy import ndarray
from numpy.testing import assert_allclose

from src.layers.embeddings.input_embedding import InputEmbedding
from src.layers.embeddings.positional_encoding import PositionalEncoding

# -----------------------------
# Fixtures
# -----------------------------


@pytest.fixture
def seed() -> int:
    return 42


@pytest.fixture
def d_model() -> int:
    return 16


@pytest.fixture
def seq_len() -> int:
    return 5


@pytest.fixture
def batch_size() -> int:
    return 2


@pytest.fixture
def vocab_size() -> int:
    return 100


@pytest.fixture
def token_input(batch_size: int, seq_len: int, vocab_size: int, seed: int) -> ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, vocab_size, size=(batch_size, seq_len))


@pytest.fixture
def embedding_layer(d_model: int, vocab_size: int, seed: int) -> InputEmbedding:
    return InputEmbedding(d_model=d_model, vocab_size=vocab_size, seed=seed)


@pytest.fixture
def pos_encoding_layer(d_model: int, seq_len: int) -> PositionalEncoding:
    return PositionalEncoding(d_model=d_model, max_len=seq_len, dropout_rate=0.0)


# -----------------------------
# Tests
# -----------------------------


def test_embedding_output_shape(
    embedding_layer: InputEmbedding, token_input: ndarray, d_model: int
) -> None:
    out = embedding_layer.forward(token_input)
    assert out.shape == (*token_input.shape, d_model)


def test_positional_encoding_output_shape(
    embedding_layer: InputEmbedding,
    pos_encoding_layer: PositionalEncoding,
    token_input: ndarray,
) -> None:
    embedded = embedding_layer.forward(token_input)
    out = pos_encoding_layer.forward(embedded)
    assert out.shape == embedded.shape


def test_positional_encoding_changes_values(
    embedding_layer: InputEmbedding,
    pos_encoding_layer: PositionalEncoding,
    token_input: ndarray,
) -> None:
    embedded = embedding_layer.forward(token_input)
    encoded = pos_encoding_layer.forward(embedded)
    # Positional encoding should change at least some values
    assert_allclose(encoded, embedded)


def test_no_nans_in_output(
    embedding_layer: InputEmbedding,
    pos_encoding_layer: PositionalEncoding,
    token_input: ndarray,
) -> None:
    embedded = embedding_layer.forward(token_input)
    encoded = pos_encoding_layer.forward(embedded)
    assert not np.isnan(encoded).any()


def test_value_range_of_encoded_output(
    embedding_layer: InputEmbedding,
    pos_encoding_layer: PositionalEncoding,
    token_input: ndarray,
) -> None:
    embedded = embedding_layer.forward(token_input)
    encoded = pos_encoding_layer.forward(embedded)
    min_val, max_val = encoded.min(), encoded.max()
    assert np.isfinite(min_val) and np.isfinite(max_val)
    assert min_val < max_val  # Sanity check for values
