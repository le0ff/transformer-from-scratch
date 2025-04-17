import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from src.layers.embeddings.input_embedding import InputEmbedding


@pytest.fixture
def d_model() -> int:
    return 8


@pytest.fixture
def vocab_size() -> int:
    return 100


@pytest.fixture
def embedding_layer(d_model: int, vocab_size: int) -> InputEmbedding:
    return InputEmbedding(d_model=d_model, vocab_size=vocab_size, seed=42)


@pytest.fixture
def token_input() -> np.ndarray:
    # shape (2, 3)
    return np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)


def test_forward_output_shape(embedding_layer: InputEmbedding, token_input: np.ndarray):
    out = embedding_layer.forward(token_input)
    assert out.shape == (2, 3, embedding_layer.d_model)


def test_forward_output_scaled(
    embedding_layer: InputEmbedding, token_input: np.ndarray
):
    out = embedding_layer.forward(token_input)
    raw = embedding_layer.W_embed[token_input]
    expected = raw * np.sqrt(embedding_layer.d_model)
    assert_allclose(out, expected, atol=1e-6)


def test_get_parameters(embedding_layer: InputEmbedding):
    params = embedding_layer.get_parameters()
    assert "W_embed" in params
    assert params["W_embed"].shape == (
        embedding_layer.vocab_size,
        embedding_layer.d_model,
    )
    assert_array_equal(params["W_embed"], embedding_layer.W_embed)


def test_set_parameters_valid(embedding_layer: InputEmbedding):
    new_weights = np.ones(
        (embedding_layer.vocab_size, embedding_layer.d_model), dtype=np.float32
    )
    embedding_layer.set_parameters({"W_embed": new_weights})
    assert_array_equal(embedding_layer.W_embed, new_weights)


@pytest.mark.parametrize(
    "x, err_msg",
    [
        (np.array([[[1]]]), "must have 2 dimensions"),
        (np.array([[1.5, 2.5]]), "must contain integer token indices"),
        (np.array([[100, 1]]), "out of bounds"),
        (np.array([[-1, 0]]), "out of bounds"),
    ],
    ids=["bad_shape", "non_integer_input", "token_id_too_high", "token_id_negative"],
)
def test_forward_invalid_inputs(embedding_layer: InputEmbedding, x, err_msg):
    with pytest.raises(ValueError, match=err_msg):
        embedding_layer.forward(x)


@pytest.mark.parametrize(
    "params, err_msg",
    [
        ({}, "Missing parameter"),
        ({"W_embed": np.zeros((1, 1))}, "Shape mismatch"),
    ],
    ids=["missing_param", "wrong_shape"],
)
def test_set_parameters_invalid(embedding_layer: InputEmbedding, params, err_msg):
    with pytest.raises(ValueError, match=err_msg):
        embedding_layer.set_parameters(params)


def test_forward_caches_input(embedding_layer: InputEmbedding, token_input: np.ndarray):
    embedding_layer.forward(token_input)
    assert hasattr(embedding_layer, "_input_cache")
    assert_array_equal(embedding_layer._input_cache, token_input)
