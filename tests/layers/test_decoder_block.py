from typing import Any, Dict, Literal

import numpy as np
import pytest
from numpy import ndarray
from numpy.testing import assert_allclose, assert_array_equal

from src.layers.decoderblock import DecoderBlock
from src.layers.feedforward import FeedForwardBlock
from src.layers.multiheadattentionblock import MultiHeadAttentionBlock

# --- Fixtures ---


@pytest.fixture
def decoderblock_config() -> Dict[str, Any]:
    return dict(d_model=16, n_heads=4, d_ff=32, dropout=0.1)


@pytest.fixture
def main_seed() -> int:
    return 42


@pytest.fixture
def decoder_block(decoderblock_config: Dict[str, Any]) -> DecoderBlock:
    self_attn = MultiHeadAttentionBlock(
        decoderblock_config["d_model"],
        decoderblock_config["n_heads"],
        decoderblock_config["dropout"],
        seed=None,
    )
    cross_attn = MultiHeadAttentionBlock(
        decoderblock_config["d_model"],
        decoderblock_config["n_heads"],
        decoderblock_config["dropout"],
        seed=None,
    )
    ffn = FeedForwardBlock(
        decoderblock_config["d_model"],
        decoderblock_config["d_ff"],
        decoderblock_config["dropout"],
        seed=None,
    )
    return DecoderBlock(
        self_attention_block=self_attn,
        cross_attention_block=cross_attn,
        feed_forward_block=ffn,
        dropout=decoderblock_config["dropout"],
        seed=None,
    )


@pytest.fixture
def decoder_block_seeded(
    decoderblock_config: Dict[str, Any], main_seed: int
) -> DecoderBlock:
    rng = np.random.default_rng(main_seed)
    seeds = rng.integers(0, 2**31 - 1, size=4)
    self_attn = MultiHeadAttentionBlock(
        decoderblock_config["d_model"],
        decoderblock_config["n_heads"],
        decoderblock_config["dropout"],
        seed=int(seeds[0]),
    )
    cross_attn = MultiHeadAttentionBlock(
        decoderblock_config["d_model"],
        decoderblock_config["n_heads"],
        decoderblock_config["dropout"],
        seed=int(seeds[1]),
    )
    ffn = FeedForwardBlock(
        decoderblock_config["d_model"],
        decoderblock_config["d_ff"],
        decoderblock_config["dropout"],
        seed=int(seeds[2]),
    )
    return DecoderBlock(
        self_attention_block=self_attn,
        cross_attention_block=cross_attn,
        feed_forward_block=ffn,
        dropout=decoderblock_config["dropout"],
        seed=int(seeds[3]),
    )


@pytest.fixture
def sample_inputs(decoderblock_config: Dict[str, Any]) -> Dict[str, ndarray]:
    batch, src_len, tgt_len = 2, 5, 3
    d_model = decoderblock_config["d_model"]
    n_heads = decoderblock_config["n_heads"]
    rng = np.random.default_rng(123)
    x = rng.standard_normal((batch, tgt_len, d_model)).astype(np.float32)
    enc_out = rng.standard_normal((batch, src_len, d_model)).astype(np.float32)
    tgt_mask = np.broadcast_to(
        np.tril(np.ones((tgt_len, tgt_len), dtype=bool)),
        (batch, n_heads, tgt_len, tgt_len),
    )
    src_mask = np.ones((batch, n_heads, tgt_len, src_len), dtype=bool)
    return {"x": x, "enc": enc_out, "tgt_mask": tgt_mask, "src_mask": src_mask}


# --- Tests ---


def test_decoderblock_forward_shape(
    decoder_block_seeded: DecoderBlock, sample_inputs: Dict[str, ndarray]
) -> None:
    out = decoder_block_seeded.forward(
        sample_inputs["x"],
        sample_inputs["enc"],
        sample_inputs["tgt_mask"],
        sample_inputs["src_mask"],
    )
    assert out.shape == sample_inputs["x"].shape


def test_decoderblock_forward_deterministic(
    decoder_block_seeded: DecoderBlock, sample_inputs: Dict[str, ndarray]
) -> None:
    decoder_block_seeded.eval()
    y1 = decoder_block_seeded.forward(
        sample_inputs["x"],
        sample_inputs["enc"],
        sample_inputs["tgt_mask"],
        sample_inputs["src_mask"],
    )
    y2 = decoder_block_seeded.forward(
        sample_inputs["x"],
        sample_inputs["enc"],
        sample_inputs["tgt_mask"],
        sample_inputs["src_mask"],
    )
    assert_array_equal(y1, y2)


def test_decoderblock_train_eval_propagation(decoder_block: DecoderBlock) -> None:
    assert decoder_block.training
    for layer in decoder_block._layers.values():
        assert layer.training
    decoder_block.eval()
    assert not decoder_block.training
    for layer in decoder_block._layers.values():
        assert not layer.training
    decoder_block.train()
    assert decoder_block.training
    for layer in decoder_block._layers.values():
        assert layer.training


def test_decoderblock_get_parameters_keys(decoder_block_seeded: DecoderBlock) -> None:
    params = decoder_block_seeded.get_parameters()
    expected_prefixes = {
        "self_attention_",
        "cross_attention_",
        "feed_forward_",
        "residual1_",
        "residual2_",
        "residual3_",
    }
    for prefix in expected_prefixes:
        assert any(k.startswith(prefix) for k in params), f"Missing prefix: {prefix}"


def test_decoderblock_get_set_roundtrip(decoder_block_seeded: DecoderBlock) -> None:
    params = decoder_block_seeded.get_parameters()
    modified = {k: v + 1.0 for k, v in params.items()}
    decoder_block_seeded.set_parameters(modified)
    returned = decoder_block_seeded.get_parameters()
    for k in params:
        assert_array_equal(returned[k], modified[k])


def test_decoderblock_seed_reproducibility(
    decoderblock_config: Dict[str, Any],
    main_seed: int,
    sample_inputs: Dict[str, ndarray],
) -> None:
    main_rng = np.random.default_rng(main_seed)
    max_seed_val = 2**31 - 1
    seeds = main_rng.integers(0, max_seed_val, size=4)
    sa_seed, ca_seed, ffn_seed, block_seed = map(int, seeds)

    self_attn1 = MultiHeadAttentionBlock(
        d_model=decoderblock_config["d_model"],
        n_heads=decoderblock_config["n_heads"],
        dropout_rate=decoderblock_config["dropout"],
        seed=sa_seed,
    )
    cross_attn1 = MultiHeadAttentionBlock(
        d_model=decoderblock_config["d_model"],
        n_heads=decoderblock_config["n_heads"],
        dropout_rate=decoderblock_config["dropout"],
        seed=ca_seed,
    )
    ffn1 = FeedForwardBlock(
        d_model=decoderblock_config["d_model"],
        d_ff=decoderblock_config["d_ff"],
        dropout=decoderblock_config["dropout"],
        seed=ffn_seed,
    )
    block1 = DecoderBlock(
        self_attention_block=self_attn1,
        cross_attention_block=cross_attn1,
        feed_forward_block=ffn1,
        dropout=decoderblock_config["dropout"],
        seed=block_seed,
    )

    self_attn2 = MultiHeadAttentionBlock(
        d_model=decoderblock_config["d_model"],
        n_heads=decoderblock_config["n_heads"],
        dropout_rate=decoderblock_config["dropout"],
        seed=sa_seed,
    )
    cross_attn2 = MultiHeadAttentionBlock(
        d_model=decoderblock_config["d_model"],
        n_heads=decoderblock_config["n_heads"],
        dropout_rate=decoderblock_config["dropout"],
        seed=ca_seed,
    )
    ffn2 = FeedForwardBlock(
        d_model=decoderblock_config["d_model"],
        d_ff=decoderblock_config["d_ff"],
        dropout=decoderblock_config["dropout"],
        seed=ffn_seed,
    )
    block2 = DecoderBlock(
        self_attention_block=self_attn2,
        cross_attention_block=cross_attn2,
        feed_forward_block=ffn2,
        dropout=decoderblock_config["dropout"],
        seed=block_seed,
    )

    block1.eval()
    block2.eval()

    out1 = block1.forward(
        sample_inputs["x"],
        sample_inputs["enc"],
        sample_inputs["tgt_mask"],
        sample_inputs["src_mask"],
    )
    out2 = block2.forward(
        sample_inputs["x"],
        sample_inputs["enc"],
        sample_inputs["tgt_mask"],
        sample_inputs["src_mask"],
    )

    assert_allclose(out1, out2, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "bad_params, error_msg",
    [
        ({"not_a_real_param": np.ones((2, 2))}, "Unexpected parameter key"),
        ({}, "No parameters provided for DecoderBlock"),
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
    decoder_block_seeded: DecoderBlock, bad_params, error_msg
) -> None:
    """Test set_parameters raises errors for invalid input."""
    with pytest.raises(ValueError, match=error_msg):
        decoder_block_seeded.set_parameters(bad_params)


@pytest.mark.parametrize(
    "dropout_val, error_msg",
    [
        (1.0, "Dropout must be a float in \\[0.0, 1.0\\)"),
        (-0.1, "Dropout must be a float in \\[0.0, 1.0\\)"),
        ("abc", "Dropout must be a float in \\[0.0, 1.0\\)"),
    ],
)
def test_decoderblock_invalid_dropout(
    decoderblock_config: Dict[str, Any],
    dropout_val: float | Literal["abc"],
    error_msg: str,
) -> None:
    sa = MultiHeadAttentionBlock(
        decoderblock_config["d_model"],
        decoderblock_config["n_heads"],
        decoderblock_config["dropout"],
        seed=None,
    )
    ca = MultiHeadAttentionBlock(
        decoderblock_config["d_model"],
        decoderblock_config["n_heads"],
        decoderblock_config["dropout"],
        seed=None,
    )
    ffn = FeedForwardBlock(
        decoderblock_config["d_model"],
        decoderblock_config["d_ff"],
        decoderblock_config["dropout"],
        seed=None,
    )
    with pytest.raises(ValueError, match=error_msg):
        DecoderBlock(sa, ca, ffn, dropout_val, None)
