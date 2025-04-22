from typing import Any, Dict, Tuple

import numpy as np
import pytest
from numpy import ndarray
from numpy.testing import assert_allclose, assert_array_equal

from src.decoder import Decoder

# --- Fixtures ---


@pytest.fixture
def decoder_config() -> Dict[str, Any]:
    """Basic Decoder configuration."""
    return {"d_model": 16, "n_heads": 4, "d_ff": 32, "dropout": 0.1, "num_blocks": 3}


@pytest.fixture
def decoder_seed() -> int:
    """Seed for reproducible tests."""
    return 123


@pytest.fixture
def decoder(decoder_config: Dict[str, Any], decoder_seed: int) -> Decoder:
    return Decoder.from_config(
        d_model=decoder_config["d_model"],
        n_heads=decoder_config["n_heads"],
        d_ff=decoder_config["d_ff"],
        dropout=decoder_config["dropout"],
        num_blocks=decoder_config["num_blocks"],
        seed=decoder_seed,
    )


@pytest.fixture
def sample_decoder_input(
    decoder_config: Dict[str, Any],
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """Sample input, encoder output, and masks for decoder."""
    batch_size = 2
    seq_len = 5
    d_model = decoder_config["d_model"]
    rng = np.random.default_rng(456)

    x = rng.standard_normal((batch_size, seq_len, d_model))
    encoder_output = rng.standard_normal((batch_size, seq_len, d_model))
    src_mask = np.ones((batch_size, decoder_config["n_heads"], seq_len, seq_len))
    tgt_mask = np.triu(np.ones_like(src_mask), 1) * -1e9

    return x, encoder_output, src_mask, tgt_mask


# --- Tests ---


def test_decoder_forward_shape(
    decoder: Decoder, sample_decoder_input: Tuple[ndarray, ndarray, ndarray, ndarray]
) -> None:
    """Test output shape matches input shape after all blocks and norm."""
    x, enc_output, src_mask, tgt_mask = sample_decoder_input
    decoder.train()
    out = decoder(x=x, encoder_output=enc_output, src_mask=src_mask, tgt_mask=tgt_mask)
    assert out.shape == x.shape


def test_decoder_train_eval_propagation(decoder: Decoder) -> None:
    """Test train/eval mode propagates to all blocks and norm."""
    assert decoder.training is True
    for block in decoder.layers.values():
        assert block.training is True
    assert decoder.norm.training is True

    decoder.eval()
    assert decoder.training is False
    for block in decoder.layers.values():
        assert block.training is False
    assert decoder.norm.training is False

    decoder.train()
    assert decoder.training is True
    for block in decoder.layers.values():
        assert block.training is True
    assert decoder.norm.training is True


def test_decoder_get_parameters_keys(decoder: Decoder) -> None:
    """Test get_parameters returns all expected keys with correct prefixes."""
    params = decoder.get_parameters()
    expected_keys = set()
    for name, layer in decoder.layers.items():
        for subkey in layer.get_parameters():
            expected_keys.add(f"{name}_{subkey}")
    for key in decoder.norm.get_parameters():
        expected_keys.add(f"norm_{key}")
    assert set(params.keys()) == expected_keys


def test_decoder_get_set_parameters_roundtrip(decoder: Decoder) -> None:
    """Test set_parameters restores parameters exactly."""
    params = decoder.get_parameters()
    new_params = {k: v + 1.0 for k, v in params.items()}
    decoder.set_parameters(new_params)
    params_after = decoder.get_parameters()
    for k in params:
        assert_array_equal(params_after[k], new_params[k])


@pytest.mark.parametrize(
    "bad_params, error_msg",
    [
        ({"not_a_real_param": np.ones((2, 2))}, "Unexpected parameter key"),
        ({}, "No parameters provided for Decoder"),
        ({"block999_invalid": np.ones((2, 2))}, "Unexpected parameter key"),
        (
            {
                "block0_self_attention_w_q": np.ones((2, 2)),
                "block1_feed_forward_w_o": np.ones((2, 2)),
            },
            "Expected 4 parameters: 'w_q', 'w_k', 'w_v', and 'w_o'",
        ),
    ],
)
def test_decoder_set_parameters_invalid(
    decoder: Decoder, bad_params, error_msg
) -> None:
    """Test set_parameters raises errors for invalid input."""
    with pytest.raises(ValueError, match=error_msg):
        decoder.set_parameters(bad_params)


def test_decoder_seed_reproducibility(
    decoder_config: Dict[str, Any],
    decoder_seed: int,
    sample_decoder_input: Tuple[ndarray, ndarray, ndarray, ndarray],
) -> None:
    """Test that same seed produces identical outputs for all blocks."""
    decoder1 = Decoder.from_config(
        d_model=decoder_config["d_model"],
        n_heads=decoder_config["n_heads"],
        d_ff=decoder_config["d_ff"],
        dropout=decoder_config["dropout"],
        num_blocks=decoder_config["num_blocks"],
        seed=decoder_seed,
    )
    decoder2 = Decoder.from_config(
        d_model=decoder_config["d_model"],
        n_heads=decoder_config["n_heads"],
        d_ff=decoder_config["d_ff"],
        dropout=decoder_config["dropout"],
        num_blocks=decoder_config["num_blocks"],
        seed=decoder_seed,
    )
    x, enc_output, src_mask, tgt_mask = sample_decoder_input
    out1 = decoder1(
        x=x, encoder_output=enc_output, src_mask=src_mask, tgt_mask=tgt_mask
    )
    out2 = decoder2(
        x=x, encoder_output=enc_output, src_mask=src_mask, tgt_mask=tgt_mask
    )
    assert_allclose(out1, out2, rtol=1e-5, atol=1e-6)

    decoder1.eval()
    decoder2.eval()
    out1_eval = decoder1(
        x=x, encoder_output=enc_output, src_mask=src_mask, tgt_mask=tgt_mask
    )
    out2_eval = decoder2(
        x=x, encoder_output=enc_output, src_mask=src_mask, tgt_mask=tgt_mask
    )
    assert_allclose(out1_eval, out2_eval, rtol=1e-5, atol=1e-6)
