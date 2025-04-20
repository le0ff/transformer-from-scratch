from typing import Any, Dict, Literal, Tuple

import numpy as np
import pytest
from numpy import ndarray
from numpy.testing import assert_allclose, assert_array_equal

from src.layers.encoderblock import EncoderBlock
from src.layers.feedforward import FeedForwardBlock
from src.layers.multiheadattentionblock import MultiHeadAttentionBlock

# --- Fixtures ---


@pytest.fixture
def encoderblock_config() -> Dict[str, Any]:
    """Basic EncoderBlock configuration."""
    return {"d_model": 16, "n_heads": 4, "d_ff": 32, "dropout": 0.1}


@pytest.fixture
def main_seed() -> int:
    """Seed for reproducible tests."""
    return 42


@pytest.fixture
def encoder_block(encoderblock_config: Dict[str, Any]) -> EncoderBlock:
    """EncoderBlock without seed."""
    attn = MultiHeadAttentionBlock(
        d_model=encoderblock_config["d_model"],
        n_heads=encoderblock_config["n_heads"],
        dropout_rate=encoderblock_config["dropout"],
        seed=None,
    )
    ff = FeedForwardBlock(
        d_model=encoderblock_config["d_model"],
        d_ff=encoderblock_config["d_ff"],
        dropout=encoderblock_config["dropout"],
        seed=None,
    )
    return EncoderBlock(
        self_attention_block=attn,
        feed_forward_block=ff,
        dropout=encoderblock_config["dropout"],
        seed=None,
    )


@pytest.fixture
def encoder_block_seeded(
    encoderblock_config: Dict[str, Any], main_seed: int
) -> EncoderBlock:
    """EncoderBlock with seed."""
    # derive seeds for each layer from main seed
    main_rng = np.random.default_rng(main_seed)
    max_seed_val = 2**31 - 1
    seeds = main_rng.integers(0, max_seed_val, size=3)
    attn_seed = int(seeds[0])
    ff_seed = int(seeds[1])
    encoderblock_seed = int(seeds[2])

    attn = MultiHeadAttentionBlock(
        d_model=encoderblock_config["d_model"],
        n_heads=encoderblock_config["n_heads"],
        dropout_rate=encoderblock_config["dropout"],
        seed=attn_seed,
    )
    ff = FeedForwardBlock(
        d_model=encoderblock_config["d_model"],
        d_ff=encoderblock_config["d_ff"],
        dropout=encoderblock_config["dropout"],
        seed=ff_seed,
    )
    return EncoderBlock(
        self_attention_block=attn,
        feed_forward_block=ff,
        dropout=encoderblock_config["dropout"],
        seed=encoderblock_seed,
    )


@pytest.fixture
def sample_input(encoderblock_config: Dict[str, Any]) -> Tuple[ndarray, ndarray]:
    """Sample input and mask for encoder block."""
    batch_size = 2
    seq_len = 5
    rng = np.random.default_rng(123)
    x = rng.standard_normal(
        (batch_size, seq_len, encoderblock_config["d_model"])
    ).astype(np.float32)
    mask = np.ones(
        (batch_size, encoderblock_config["n_heads"], seq_len, seq_len), dtype=np.float32
    )
    return x, mask


# --- Test Functions ---


def test_encoderblock_forward_shape(
    encoder_block_seeded: EncoderBlock, sample_input: Tuple[ndarray, ndarray]
) -> None:
    """Test output shape matches input shape."""
    x, mask = sample_input
    encoder_block_seeded.train()
    out = encoder_block_seeded(x, mask=mask)
    assert out.shape == x.shape


def test_encoderblock_train_eval_propagation(encoder_block: EncoderBlock) -> None:
    """Test train/eval mode propagates to all sublayers."""
    # Initial state (should be training)
    assert encoder_block.training is True
    for layer in encoder_block._layers.values():
        assert layer.training is True

    encoder_block.eval()
    assert encoder_block.training is False
    for layer in encoder_block._layers.values():
        assert layer.training is False

    encoder_block.train()
    assert encoder_block.training is True
    for layer in encoder_block._layers.values():
        assert layer.training is True


def test_encoderblock_get_parameters_keys(encoder_block_seeded: EncoderBlock) -> None:
    """Test get_parameters returns expected keys."""
    params = encoder_block_seeded.get_parameters()
    expected_prefixes = {"self_attention_", "feed_forward_", "residual1_", "residual2_"}
    for prefix in expected_prefixes:
        assert any(k.startswith(prefix) for k in params), f"Missing prefix: {prefix}"


def test_encoderblock_get_set_parameters_roundtrip(
    encoder_block_seeded: EncoderBlock,
) -> None:
    """Test set_parameters restores parameters exactly."""
    params = encoder_block_seeded.get_parameters()
    # Modify parameters
    new_params = {k: v + 1.0 for k, v in params.items()}
    # set modified parameters
    encoder_block_seeded.set_parameters(new_params)
    params_after = encoder_block_seeded.get_parameters()
    for k in params:
        assert_array_equal(params_after[k], new_params[k])


@pytest.mark.parametrize(
    "bad_params, error_msg",
    [
        ({"not_a_real_param": np.ones((2, 2))}, "Unexpected parameter key"),
        ({}, "No parameters provided for EncoderBlock"),
        (
            {
                "self_attention_w_q": np.ones((2, 2)),
                "feed_forward_w_o": np.ones((2, 2)),
            },
            "Expected 4 parameters: 'w_q', 'w_k', 'w_v', and 'w_o'",
        ),
    ],
)
def test_encoderblock_set_parameters_invalid(
    encoder_block_seeded: EncoderBlock, bad_params, error_msg
) -> None:
    """Test set_parameters raises errors for invalid input."""
    with pytest.raises(ValueError, match=error_msg):
        encoder_block_seeded.set_parameters(bad_params)


def test_encoderblock_seed_reproducibility(
    encoderblock_config: Dict[str, Any],
    main_seed: int,
    sample_input: Tuple[ndarray, ndarray],
) -> None:
    """Test that same seed produces identical outputs."""
    main_rng = np.random.default_rng(main_seed)
    max_seed_val = 2**31 - 1
    seeds = main_rng.integers(0, max_seed_val, size=3)
    attn_seed = int(seeds[0])
    ff_seed = int(seeds[1])
    encoderblock_seed = int(seeds[2])

    attn1 = MultiHeadAttentionBlock(
        d_model=encoderblock_config["d_model"],
        n_heads=encoderblock_config["n_heads"],
        dropout_rate=encoderblock_config["dropout"],
        seed=attn_seed,
    )
    ff1 = FeedForwardBlock(
        d_model=encoderblock_config["d_model"],
        d_ff=encoderblock_config["d_ff"],
        dropout=encoderblock_config["dropout"],
        seed=ff_seed,
    )
    block1 = EncoderBlock(attn1, ff1, encoderblock_config["dropout"], encoderblock_seed)

    attn2 = MultiHeadAttentionBlock(
        d_model=encoderblock_config["d_model"],
        n_heads=encoderblock_config["n_heads"],
        dropout_rate=encoderblock_config["dropout"],
        seed=attn_seed,
    )
    ff2 = FeedForwardBlock(
        d_model=encoderblock_config["d_model"],
        d_ff=encoderblock_config["d_ff"],
        dropout=encoderblock_config["dropout"],
        seed=ff_seed,
    )
    block2 = EncoderBlock(attn2, ff2, encoderblock_config["dropout"], encoderblock_seed)

    x, mask = sample_input
    block1.eval()
    block2.eval()
    out1 = block1(x, mask=mask)
    out2 = block2(x, mask=mask)
    assert_allclose(out1, out2, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "dropout_val, error_msg",
    [
        (1.0, "Dropout must be a float in \\[0.0, 1.0\\)"),
        (-0.1, "Dropout must be a float in \\[0.0, 1.0\\)"),
        ("abc", "Dropout must be a float in \\[0.0, 1.0\\)"),
    ],
)
def test_encoderblock_invalid_dropout(
    encoderblock_config: Dict[str, Any],
    dropout_val: float | Literal["abc"],
    error_msg: Literal["Dropout must be a float in \\[0.0, 1.0\\)"],
) -> None:
    """Test invalid dropout values raise ValueError."""
    attn = MultiHeadAttentionBlock(
        d_model=encoderblock_config["d_model"],
        n_heads=encoderblock_config["n_heads"],
        dropout_rate=encoderblock_config["dropout"],
        seed=None,
    )
    ff = FeedForwardBlock(
        d_model=encoderblock_config["d_model"],
        d_ff=encoderblock_config["d_ff"],
        dropout=encoderblock_config["dropout"],
        seed=None,
    )
    with pytest.raises(ValueError, match=error_msg):
        EncoderBlock(attn, ff, dropout_val, None)
