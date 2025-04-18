import numpy as np
import pytest
from numpy import ndarray
from numpy.testing import assert_allclose, assert_array_equal

from src.layers.embeddings.positional_encoding import PositionalEncoding


@pytest.fixture
def d_model() -> int:
    return 16


@pytest.fixture
def max_len() -> int:
    return 100


@pytest.fixture
def dropout_rate() -> float:
    return 0.5


@pytest.fixture
def pe_block(d_model: int, max_len: int, dropout_rate: float) -> PositionalEncoding:
    return PositionalEncoding(
        d_model=d_model, max_len=max_len, dropout_rate=dropout_rate
    )


@pytest.fixture
def sample_input(d_model: int) -> ndarray:
    rng = np.random.default_rng(123)
    return rng.standard_normal((2, 10, d_model)).astype(np.float32)


def test_forward_shape(pe_block: PositionalEncoding, sample_input: ndarray):
    """Output shape should == input shape."""
    out = pe_block.forward(sample_input)
    assert out.shape == sample_input.shape


def test_forward_adds_encoding(pe_block: PositionalEncoding, sample_input: ndarray):
    """Check that positional encoding is actually added."""
    out = pe_block.forward(sample_input)
    assert not np.allclose(out, sample_input), "Positional encoding should alter input."


def test_pe_consistency(pe_block: PositionalEncoding):
    """Same positional encoding is applied each time (deterministic)."""
    pe1 = pe_block.pe.copy()
    pe2 = pe_block.build_pe(pe_block.max_len, pe_block.d_model)
    assert_array_equal(pe1, pe2)


def test_encoding_for_known_position(pe_block: PositionalEncoding):
    """Check known sin/cos structure for position 0 (should be sin(0)=0, cos(0)=1)."""
    first_pos = pe_block.pe[0]
    assert_allclose(first_pos[0::2], 0.0, atol=1e-6)
    assert_allclose(first_pos[1::2], 1.0, atol=1e-6)


@pytest.mark.parametrize(
    "bad_d_model, bad_max_len, error_msg",
    [
        (0, 100, "d_model must be a positive integer"),
        (-4, 100, "d_model must be a positive integer"),
        (16, 0, "max_len must be a positive integer"),
        (16, -1, "max_len must be a positive integer"),
    ],
    ids=["zero_d_model", "neg_d_model", "zero_max_len", "neg_max_len"],
)
def test_invalid_params(bad_d_model, bad_max_len, error_msg):
    """Test if invalid parameter values raise ValueError."""
    with pytest.raises(ValueError, match=error_msg):
        PositionalEncoding(d_model=bad_d_model, max_len=bad_max_len, dropout_rate=0.0)


@pytest.mark.parametrize("shape", [(2, 10, 16)], ids=["3d"])
def test_forward_shape_flexible(pe_block: PositionalEncoding, shape):
    dummy_input = np.random.randn(*shape).astype(np.float32)
    out = pe_block.forward(dummy_input)
    assert out.shape == dummy_input.shape
