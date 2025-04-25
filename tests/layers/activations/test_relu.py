import numpy as np
import pytest
from numpy import ndarray

from src.layers.activations.relu import ReLU


@pytest.fixture
def relu_layer() -> ReLU:
    """Creates an instance of the ReLU activation layer."""
    return ReLU()


@pytest.fixture
def sample_input_mixed_signs() -> ndarray:
    """Creates sample input data with positive, negative, and zero values."""
    # Shape: (batch_size, features)
    data = np.array([[1.0, -2.0, 0.0, 3.5, -0.1], [-5.0, 6.0, 0.0, -0.0, 10.0]])
    return data


@pytest.fixture
def sample_input_mixed_signs_expected() -> ndarray:
    """Creates expected output data for the sample input."""
    # Shape: (batch_size, features)
    data = np.array([[1.0, 0.0, 0.0, 3.5, 0.0], [0.0, 6.0, 0.0, 0.0, 10.0]])
    return data


@pytest.fixture
def sample_input_3d() -> ndarray:
    """Creates sample 3D input data."""
    # Shape: (batch_size, seq_len, features)
    return np.random.randn(4, 8, 10)


def test_relu_get_parameters(relu_layer: ReLU) -> None:
    """Tests that ReLU layer has no parameters."""
    params = relu_layer.get_parameters()
    assert isinstance(params, dict), "get_parameters should return a dictionary"
    assert len(params) == 0, "ReLU layer should have no parameters"


def test_relu_forward_shape(
    relu_layer: ReLU, sample_input_mixed_signs: ndarray, sample_input_3d: ndarray
) -> None:
    """Tests that the forward pass preserves input shape for various dimensions."""
    # Test 2D input
    output_2d = relu_layer(sample_input_mixed_signs)
    assert output_2d.shape == sample_input_mixed_signs.shape, (
        f"Expected 2D output shape {sample_input_mixed_signs.shape}, got {output_2d.shape}"
    )

    # Test 3D input
    output_3d = relu_layer(sample_input_3d)
    assert output_3d.shape == sample_input_3d.shape, (
        f"Expected 3D output shape {sample_input_3d.shape}, got {output_3d.shape}"
    )


def test_relu_forward_computation(
    relu_layer: ReLU,
    sample_input_mixed_signs: ndarray,
    sample_input_mixed_signs_expected: ndarray,
) -> None:
    """Tests the ReLU computation max(0, x)."""
    x = sample_input_mixed_signs
    expected_output = sample_input_mixed_signs_expected
    actual_output = relu_layer(x)

    # ReLU computation should be exact
    np.testing.assert_array_equal(
        actual_output, expected_output, err_msg="ReLU forward pass computation mismatch"
    )


def test_relu_forward_input_cache(
    relu_layer: ReLU, sample_input_mixed_signs: ndarray
) -> None:
    """Tests if the input is cached correctly during the forward pass."""
    assert relu_layer._input_cache is None, (
        "Input cache should be None before forward pass"
    )
    relu_layer(sample_input_mixed_signs)
    assert relu_layer._input_cache is not None, (
        "Input cache should not be None after forward pass"
    )
    np.testing.assert_array_equal(
        relu_layer._input_cache,
        sample_input_mixed_signs,
        err_msg="Cached input does not match actual input",
    )
