import numpy as np
import pytest
from numpy import ndarray
from numpy.testing import assert_allclose, assert_array_equal

from src.layers.normalization import LayerNorm


@pytest.fixture
def norm_shape() -> int:
    return 8


@pytest.fixture
def sample_input_2d(norm_shape: int) -> ndarray:
    rng = np.random.default_rng(123)
    return rng.standard_normal((4, norm_shape)).astype(np.float32)


@pytest.fixture
def sample_input_3d(norm_shape: int) -> ndarray:
    rng = np.random.default_rng(123)
    return rng.standard_normal((2, 3, norm_shape)).astype(np.float32)


@pytest.fixture
def layer_norm(norm_shape: int) -> LayerNorm:
    return LayerNorm(normalized_shape=norm_shape)


def test_forward_output_shape_2d(layer_norm: LayerNorm, sample_input_2d: ndarray):
    """Test output shape matches input shape for 2D input."""
    out = layer_norm.forward(sample_input_2d)
    assert out.shape == sample_input_2d.shape


def test_forward_output_shape_3d(layer_norm: LayerNorm, sample_input_3d: ndarray):
    """Test output shape matches input shape for 3D input."""
    out = layer_norm.forward(sample_input_3d)
    assert out.shape == sample_input_3d.shape


def test_forward_normalization_mean_std(
    layer_norm: LayerNorm, sample_input_2d: ndarray
):
    """Test that output has mean ~0 and std ~1 across last dim."""
    out = layer_norm.forward(sample_input_2d)
    mean = np.mean(out, axis=-1)
    std = np.std(out, axis=-1)

    assert_allclose(mean, 0.0, atol=1e-5)
    assert_allclose(std, 1.0, atol=1e-5)


def test_get_parameters(layer_norm: LayerNorm):
    """Test get_parameters returns gamma and beta with correct shape."""
    params = layer_norm.get_parameters()
    assert "gamma" in params and "beta" in params
    assert params["gamma"].shape == (layer_norm.normalized_shape,)
    assert params["beta"].shape == (layer_norm.normalized_shape,)
    assert_array_equal(params["gamma"], np.ones_like(params["gamma"]))
    assert_array_equal(params["beta"], np.zeros_like(params["beta"]))


def test_set_parameters_valid(layer_norm: LayerNorm):
    """Test setting parameters works with correct shapes."""
    new_gamma = np.full((layer_norm.normalized_shape,), 2.0, dtype=np.float32)
    new_beta = np.full((layer_norm.normalized_shape,), 3.0, dtype=np.float32)

    layer_norm.set_parameters({"gamma": new_gamma, "beta": new_beta})

    assert_array_equal(layer_norm.gamma, new_gamma)
    assert_array_equal(layer_norm.beta, new_beta)


@pytest.mark.parametrize(
    "invalid_params, error_msg",
    [
        ({"gamma": np.zeros((5,)), "beta": np.zeros((8,))}, "Expected gamma shape"),
        ({"gamma": np.zeros((8,)), "beta": np.zeros((5,))}, "Expected beta shape"),
        ({"beta": np.zeros((8,))}, "Scale parameter 'gamma' missing"),
        ({"gamma": np.zeros((8,))}, "Bias parameter 'beta' missing"),
    ],
    ids=[
        "wrong_gamma_shape",
        "wrong_beta_shape",
        "missing_gamma",
        "missing_beta",
    ],
)
def test_set_parameters_invalid(layer_norm: LayerNorm, invalid_params, error_msg):
    """Test that incorrect parameter shapes or missing keys raise errors."""
    with pytest.raises(ValueError, match=error_msg):
        layer_norm.set_parameters(invalid_params)


def test_forward_zero_input(layer_norm: LayerNorm, norm_shape: int):
    input_data = np.zeros((4, norm_shape), dtype=np.float32)
    output = layer_norm.forward(input_data)
    mean = output.mean(axis=-1)
    std = output.std(axis=-1)
    assert_allclose(mean, 0.0, atol=1e-5)
    assert_allclose(std, 0.0, atol=1e-5)
