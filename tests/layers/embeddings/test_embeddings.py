from typing import Any, Dict

import numpy as np
import pytest

from src.layers.embeddings.input_embedding import InputEmbedding
from src.layers.embeddings.positional_encoding import PositionalEncoding

# --- Fixtures ---


@pytest.fixture
def model_conf() -> Dict[str, Any]:
    seed = 42
    d_model = 16
    seq_len = 5
    batch_size = 2
    vocab_size = 100

    rng = np.random.default_rng(seed)
    token_input = rng.integers(0, vocab_size, size=(batch_size, seq_len))

    return {
        "seed": seed,
        "d_model": d_model,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "vocab_size": vocab_size,
        "token_input": token_input,
    }


@pytest.fixture
def embedding_layer(model_conf: Dict[str, Any]) -> InputEmbedding:
    return InputEmbedding(
        d_model=model_conf["d_model"],
        vocab_size=model_conf["vocab_size"],
        seed=model_conf["seed"],
    )


@pytest.fixture
def pos_encoding_layer(model_conf: Dict[str, Any]) -> PositionalEncoding:
    return PositionalEncoding(
        d_model=model_conf["d_model"],
        max_len=model_conf["seq_len"],
        dropout_rate=0.0,
    )


# --- Tests ---


def test_embedding_output_shape(
    embedding_layer: InputEmbedding, model_conf: Dict[str, Any]
) -> None:
    x = model_conf["token_input"]
    d_model = model_conf["d_model"]
    out = embedding_layer.forward(x)
    assert out.shape == (*x.shape, d_model)


def test_positional_encoding_output_shape(
    embedding_layer: InputEmbedding,
    pos_encoding_layer: PositionalEncoding,
    model_conf: Dict[str, Any],
) -> None:
    x = model_conf["token_input"]
    embedded = embedding_layer.forward(x)
    out = pos_encoding_layer.forward(embedded)
    assert out.shape == embedded.shape


def test_positional_encoding_changes_values(
    embedding_layer: InputEmbedding,
    pos_encoding_layer: PositionalEncoding,
    model_conf: Dict[str, Any],
) -> None:
    x = model_conf["token_input"]
    embedded = embedding_layer.forward(x)
    encoded = pos_encoding_layer.forward(embedded)
    assert not np.allclose(encoded, embedded)


def test_no_nans_in_output(
    embedding_layer: InputEmbedding,
    pos_encoding_layer: PositionalEncoding,
    model_conf: Dict[str, Any],
) -> None:
    x = model_conf["token_input"]
    embedded = embedding_layer.forward(x)
    encoded = pos_encoding_layer.forward(embedded)
    assert not np.isnan(encoded).any()


def test_value_range_of_encoded_output(
    embedding_layer: InputEmbedding,
    pos_encoding_layer: PositionalEncoding,
    model_conf: Dict[str, Any],
) -> None:
    x = model_conf["token_input"]
    embedded = embedding_layer.forward(x)
    encoded = pos_encoding_layer.forward(embedded)
    min_val, max_val = encoded.min(), encoded.max()
    assert np.isfinite(min_val) and np.isfinite(max_val)
    # Sanity check
    assert min_val < max_val
