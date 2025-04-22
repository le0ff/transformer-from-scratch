from typing import Any, Dict, Tuple

import numpy as np
import pytest
from numpy import ndarray
from numpy.testing import assert_allclose, assert_array_equal

from src.encoder import Encoder

# --- Fixtures ---


@pytest.fixture
def encoder_config() -> Dict[str, Any]:
    """Basic Encoder configuration."""
    return {"d_model": 16, "n_heads": 4, "d_ff": 32, "dropout": 0.1, "num_blocks": 3}


@pytest.fixture
def encoder_seed() -> int:
    """Seed for reproducible tests."""
    return 123


@pytest.fixture
def encoder(encoder_config: Dict[str, Any], encoder_seed: int) -> Encoder:
    """Encoder built from config and seed."""
    return Encoder.from_config(
        d_model=encoder_config["d_model"],
        n_heads=encoder_config["n_heads"],
        d_ff=encoder_config["d_ff"],
        dropout=encoder_config["dropout"],
        num_blocks=encoder_config["num_blocks"],
        seed=encoder_seed,
    )


@pytest.fixture
def sample_encoder_input(encoder_config: Dict[str, Any]) -> Tuple[ndarray, ndarray]:
    """Sample input and mask for encoder."""
    batch_size = 2
    seq_len = 5
    rng = np.random.default_rng(456)
    x = rng.standard_normal(
        (batch_size, seq_len, encoder_config["d_model"])
    )  # .astype(np.float32)

    mask = np.ones(
        (batch_size, encoder_config["n_heads"], seq_len, seq_len)  # , dtype=np.float32
    )
    return x, mask


# --- Test Functions ---


def test_encoder_forward_shape(
    encoder: Encoder, sample_encoder_input: Tuple[ndarray, ndarray]
) -> None:
    """Test output shape matches input shape after all blocks and norm."""
    x, mask = sample_encoder_input
    encoder.train()
    out = encoder(x, mask=mask)
    assert out.shape == x.shape


def test_encoder_train_eval_propagation(encoder: Encoder) -> None:
    """Test train/eval mode propagates to all blocks and norm."""
    assert encoder.training is True
    for block in encoder.layers.values():
        assert block.training is True
    assert encoder.norm.training is True

    encoder.eval()
    assert encoder.training is False
    for block in encoder.layers.values():
        assert block.training is False
    assert encoder.norm.training is False

    encoder.train()
    assert encoder.training is True
    for block in encoder.layers.values():
        assert block.training is True
    assert encoder.norm.training is True


def test_encoder_get_parameters_keys(encoder: Encoder) -> None:
    """Test get_parameters returns all expected keys with correct prefixes."""
    params = encoder.get_parameters()
    expected_keys = set()
    # Compose expected keys from all blocks and norm
    for block_name, block in encoder.layers.items():
        for sublayer_name, sublayer in block._layers.items():
            for key in sublayer.get_parameters():
                expected_keys.add(f"{block_name}_{sublayer_name}_{key}")
    # Add norm keys
    for key in encoder.norm.get_parameters():
        expected_keys.add(f"norm_{key}")
    assert set(params.keys()) == expected_keys, (
        f"Parameter keys mismatch.\nExpected: {expected_keys}\nGot: {set(params.keys())}"
    )


def test_encoder_get_set_parameters_roundtrip(encoder: Encoder) -> None:
    """Test set_parameters restores parameters exactly."""
    params = encoder.get_parameters()
    # Modify parameters
    new_params = {k: v + 1.0 for k, v in params.items()}
    encoder.set_parameters(new_params)
    params_after = encoder.get_parameters()
    for k in params:
        assert_array_equal(params_after[k], new_params[k])


@pytest.mark.parametrize(
    "bad_params, error_msg",
    [
        ({"not_a_real_param": np.ones((2, 2))}, "Unexpected parameter key"),
        ({}, "No parameters provided for Encoder"),
        (
            {
                "block0_self_attention_w_q": np.ones((2, 2)),
                "block1_feed_forward_w_o": np.ones((2, 2)),
            },
            "Expected 4 parameters: 'w_q', 'w_k', 'w_v', and 'w_o'",
        ),
    ],
)
def test_encoder_set_parameters_invalid(
    encoder: Encoder, bad_params, error_msg
) -> None:
    """Test set_parameters raises errors for invalid input."""
    with pytest.raises(ValueError, match=error_msg):
        encoder.set_parameters(bad_params)


def test_encoder_seed_reproducibility(
    encoder_config: Dict[str, Any],
    encoder_seed: int,
    sample_encoder_input: Tuple[ndarray, ndarray],
) -> None:
    """Test that same seed produces identical outputs for all blocks."""
    encoder1 = Encoder.from_config(
        d_model=encoder_config["d_model"],
        n_heads=encoder_config["n_heads"],
        d_ff=encoder_config["d_ff"],
        dropout=encoder_config["dropout"],
        num_blocks=encoder_config["num_blocks"],
        seed=encoder_seed,
    )
    encoder2 = Encoder.from_config(
        d_model=encoder_config["d_model"],
        n_heads=encoder_config["n_heads"],
        d_ff=encoder_config["d_ff"],
        dropout=encoder_config["dropout"],
        num_blocks=encoder_config["num_blocks"],
        seed=encoder_seed,
    )
    x, mask = sample_encoder_input
    out1 = encoder1(x, mask=mask)
    out2 = encoder2(x, mask=mask)
    assert_allclose(out1, out2, rtol=1e-5, atol=1e-6)
    encoder1.eval()
    encoder2.eval()
    out1_eval = encoder1(x, mask=mask)
    out2_eval = encoder2(x, mask=mask)
    assert_allclose(out1_eval, out2_eval, rtol=1e-5, atol=1e-6)
