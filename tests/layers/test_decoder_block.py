import re
from typing import Dict

import numpy as np
import pytest
from numpy.testing import assert_array_equal

# --------------------------------------------------------------------------- #
# Imports from your code-base (adjust paths if needed)
# --------------------------------------------------------------------------- #
from src.layers.decoder import DecoderBlock
from src.layers.feedforward import FeedForwardBlock
from src.layers.multiheadattentionblock import MultiHeadAttentionBlock


# --------------------------------------------------------------------------- #
# ---------------------------  Base configuration  -------------------------- #
# --------------------------------------------------------------------------- #
@pytest.fixture
def dec_cfg() -> Dict:
    """Common hyper-params for the decoder block."""
    return dict(d_model=16, n_heads=4, d_ff=32, dropout=0.1)


@pytest.fixture
def seeds() -> Dict[str, int]:
    """Deterministic seeds for reproducible tests."""
    return dict(dec=123, self_attn=1, cross_attn=2, ffn=3)


# --------------------------------------------------------------------------- #
# ------------------------------  Sublayers  -------------------------------- #
# --------------------------------------------------------------------------- #
@pytest.fixture
def self_attn(dec_cfg: Dict, seeds: Dict[str, int]) -> MultiHeadAttentionBlock:
    return MultiHeadAttentionBlock(
        d_model=dec_cfg["d_model"],
        n_heads=dec_cfg["n_heads"],
        dropout_rate=dec_cfg["dropout"],
        seed=seeds["self_attn"],
    )


@pytest.fixture
def cross_attn(dec_cfg: Dict, seeds: Dict[str, int]) -> MultiHeadAttentionBlock:
    return MultiHeadAttentionBlock(
        d_model=dec_cfg["d_model"],
        n_heads=dec_cfg["n_heads"],
        dropout_rate=dec_cfg["dropout"],
        seed=seeds["cross_attn"],
    )


@pytest.fixture
def ffn(dec_cfg: Dict, seeds: Dict[str, int]) -> FeedForwardBlock:
    return FeedForwardBlock(
        d_model=dec_cfg["d_model"],
        d_ff=dec_cfg["d_ff"],
        dropout=dec_cfg["dropout"],
        seed=seeds["ffn"],
    )


# --------------------------------------------------------------------------- #
# ----------------------------  Decoder block  ------------------------------ #
# --------------------------------------------------------------------------- #
@pytest.fixture
def decoder_block(
    self_attn: MultiHeadAttentionBlock,
    cross_attn: MultiHeadAttentionBlock,
    ffn: FeedForwardBlock,
    dec_cfg: Dict,
    seeds: Dict[str, int],
) -> DecoderBlock:
    return DecoderBlock(
        self_attention_block=self_attn,
        cross_attention_block=cross_attn,
        feed_forward_block=ffn,
        dropout=dec_cfg["dropout"],
        seed=seeds["dec"],
    )


# --------------------------------------------------------------------------- #
# --------------------------  Sample input data  ---------------------------- #
# --------------------------------------------------------------------------- #
@pytest.fixture
def sample_inputs(dec_cfg: Dict) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(999)
    batch, src_len, tgt_len = 2, 5, 3
    d_model = dec_cfg["d_model"]
    n_heads = dec_cfg["n_heads"]

    x = rng.standard_normal((batch, tgt_len, d_model)).astype(np.float32)
    enc_out = rng.standard_normal((batch, src_len, d_model)).astype(np.float32)

    # causal mask broadcast to heads (batch, heads, tgt_len, tgt_len)
    causal_base = np.tril(np.ones((tgt_len, tgt_len), bool))
    causal = np.broadcast_to(causal_base, (batch, n_heads, tgt_len, tgt_len))

    # padding mask for cross-attn (batch, heads, tgt_len, src_len)
    pad = np.ones((batch, n_heads, tgt_len, src_len), bool)

    return dict(x=x, enc=enc_out, tgt_mask=causal, src_mask=pad)


# --------------------------------------------------------------------------- #
# ------------------------------  Test cases  ------------------------------- #
# --------------------------------------------------------------------------- #
def test_forward_shape(decoder_block: DecoderBlock, sample_inputs: Dict, dec_cfg: Dict):
    y = decoder_block.forward(
        sample_inputs["x"],
        sample_inputs["enc"],
        sample_inputs["tgt_mask"],
        sample_inputs["src_mask"],
    )
    assert y.shape == sample_inputs["x"].shape
    assert y.dtype == np.float32


def test_train_eval_propagation(decoder_block: DecoderBlock):
    # start in training mode
    assert decoder_block.training
    assert decoder_block.self_attention_block.training

    decoder_block.eval()
    assert not decoder_block.training
    assert not decoder_block.self_attention_block.training
    assert not decoder_block.feed_forward_block.training

    decoder_block.train()
    assert decoder_block.training
    assert decoder_block.cross_attention_block.training


def test_get_set_roundtrip(decoder_block: DecoderBlock):
    params = decoder_block.get_parameters()
    params_mod = {k: v + 1.0 for k, v in params.items()}
    decoder_block.set_parameters(params_mod)
    params_back = decoder_block.get_parameters()

    for k in params_mod:
        assert_array_equal(params_back[k], params_mod[k])


@pytest.mark.parametrize(
    "invalid, err_re",
    [
        # Wrong dropout
        (dict(dropout=-0.1), re.escape("dropout must be in [0.0, 1.0).")),
        (dict(dropout=1.3), re.escape("dropout must be in [0.0, 1.0).")),
        # Seed not int
        (dict(seed="abc"), re.escape("seed must be an int or None.")),
    ],
    ids=["neg_dropout", "high_dropout", "seed_str"],
)
def test_init_invalid(
    invalid: Dict,
    dec_cfg: Dict,
    self_attn,
    cross_attn,
    ffn,
    err_re: str,
):
    good_kwargs = dict(
        self_attention_block=self_attn,
        cross_attention_block=cross_attn,
        feed_forward_block=ffn,
        dropout=dec_cfg["dropout"],
        seed=None,
    )
    good_kwargs.update(invalid)

    with pytest.raises(ValueError, match=err_re):
        DecoderBlock(**good_kwargs)


def test_seed_reproducibility(
    dec_cfg: Dict, seeds: Dict[str, int], self_attn, cross_attn, ffn
):
    # Same seed → identical parameters
    block1 = DecoderBlock(self_attn, cross_attn, ffn, dec_cfg["dropout"], seed=777)
    block2 = DecoderBlock(self_attn, cross_attn, ffn, dec_cfg["dropout"], seed=777)

    p1 = block1.get_parameters()
    p2 = block2.get_parameters()

    for k in p1:
        assert_array_equal(p1[k], p2[k])

    # Different main seed does not change parameters
    block3 = DecoderBlock(self_attn, cross_attn, ffn, dec_cfg["dropout"], seed=778)
    p3 = block3.get_parameters()

    for k in p1:
        assert_array_equal(p1[k], p3[k])


@pytest.mark.parametrize(
    "bad_params, err_msg",
    [
        ({"self_attn_w_q": np.zeros((1, 1))}, "Error setting parameters"),
        ({"unknownkey": np.zeros(1)}, "Unrecognised parameter key"),
    ],
    ids=["wrong_shape", "unknown_prefix"],
)
def test_set_parameters_invalid(decoder_block: DecoderBlock, bad_params, err_msg):
    with pytest.raises(ValueError, match=err_msg):
        decoder_block.set_parameters(bad_params)
