from typing import Dict, Tuple

import numpy as np
import pytest
from numpy import ndarray
from numpy.testing import assert_allclose, assert_array_equal

from src.transformer import Transformer


# --- Fixtures ---
@pytest.fixture
def transformer_config() -> Dict[str, int | float]:
    return dict(
        src_vocab_size=32,
        tgt_vocab_size=32,
        src_seq_len=8,
        tgt_seq_len=8,
        d_model=16,
        n_blocks=2,
        n_heads=4,
        dropout_rate=0.1,
        d_ff=32,
        seed=123,
    )


@pytest.fixture
def transformer(transformer_config: Dict[str, int | float]) -> Transformer:
    return Transformer.build_transformer(**transformer_config)


@pytest.fixture
def sample_batch(
    transformer_config: Dict[str, int | float],
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    batch_size = 4
    seq_len = transformer_config["src_seq_len"]
    vocab_size = transformer_config["src_vocab_size"]
    rng = np.random.default_rng(42)
    src = rng.integers(0, vocab_size, size=(batch_size, seq_len))
    tgt = rng.integers(0, vocab_size, size=(batch_size, seq_len))
    src_mask = np.ones((batch_size, transformer_config["n_heads"], seq_len, seq_len))
    tgt_mask = np.ones((batch_size, transformer_config["n_heads"], seq_len, seq_len))
    return src, tgt, src_mask, tgt_mask


# --- Tests ---
def test_transformer_init(transformer: Transformer) -> None:
    assert isinstance(transformer, Transformer)
    assert hasattr(transformer, "encoder")
    assert hasattr(transformer, "decoder")
    assert hasattr(transformer, "src_embed")
    assert hasattr(transformer, "tgt_embed")
    assert hasattr(transformer, "src_pos")
    assert hasattr(transformer, "tgt_pos")
    assert hasattr(transformer, "projection_layer")


def test_transformer_forward_shape(
    transformer: Transformer, sample_batch: Tuple[ndarray, ndarray, ndarray, ndarray]
) -> None:
    src, tgt, src_mask, tgt_mask = sample_batch
    out = transformer(src, tgt, src_mask, tgt_mask)
    # Output shape: (batch, seq_len, tgt_vocab_size)
    assert out.shape == (
        src.shape[0],
        tgt.shape[1],
        transformer.projection_layer.vocab_size,
    )


def test_transformer_train_eval_propagation(transformer: Transformer) -> None:
    # Default is training
    assert transformer.training is True
    for layer in transformer._layers.values():
        assert layer.training is True
    transformer.eval()
    assert transformer.training is False
    for layer in transformer._layers.values():
        assert layer.training is False
    transformer.train()
    assert transformer.training is True
    for layer in transformer._layers.values():
        assert layer.training is True


def test_transformer_get_parameters_keys(transformer: Transformer) -> None:
    params = transformer.get_parameters()
    # All keys should be prefixed by submodule name
    for key in params:
        assert any(key.startswith(f"{name}_") for name in transformer._layers)


def test_transformer_get_set_parameters_roundtrip(transformer: Transformer) -> None:
    params = transformer.get_parameters()
    # Modify parameters
    new_params = {k: v + 1.0 for k, v in params.items()}
    transformer.set_parameters(new_params)
    params_after = transformer.get_parameters()
    for k in params:
        assert_array_equal(params_after[k], new_params[k])


@pytest.mark.parametrize(
    "bad_params, error_msg",
    [
        ({"not_a_real_param": np.ones((2, 2))}, "Unexpected parameter key"),
        ({}, "No parameters provided for Transformer"),
    ],
)
def test_transformer_set_parameters_invalid(
    transformer: Transformer, bad_params, error_msg
):
    with pytest.raises(ValueError, match=error_msg):
        transformer.set_parameters(bad_params)


def test_transformer_seed_reproducibility(
    transformer_config: Dict[str, int | float],
    sample_batch: Tuple[ndarray, ndarray, ndarray, ndarray],
) -> None:
    t1 = Transformer.build_transformer(**transformer_config)
    t2 = Transformer.build_transformer(**transformer_config)
    src, tgt, src_mask, tgt_mask = sample_batch
    out1 = t1(src, tgt, src_mask, tgt_mask)
    out2 = t2(src, tgt, src_mask, tgt_mask)
    assert_allclose(out1, out2, rtol=1e-5, atol=1e-6)
