import numpy as np
import pytest
from numpy import ndarray

from src.layers.linear import Linear


@pytest.fixture
def layer_config() -> dict[str, int]:
    return {"input_dim": 10, "output_dim": 5}


@pytest.fixture
def linear_layer(layer_config: dict[str, int]) -> Linear:
    return Linear(layer_config["input_dim"], layer_config["output_dim"])


@pytest.fixture
def sample_input_2d(layer_config: dict[str, int]) -> np.ndarray:
    batch_size = 4
    return np.random.randn(batch_size, layer_config["input_dim"])


@pytest.fixture
def sample_input_3d(layer_config: dict[str, int]) -> np.ndarray:
    batch_size = 4
    seq_len = 8
    return np.random.randn(batch_size, seq_len, layer_config["input_dim"])


def test_linear_init(linear_layer: Linear, layer_config: dict[str, int]) -> None:
    """Tests layer initialiaztion: dimensions and default mode."""
    input_dim = layer_config["input_dim"]
    output_dim = layer_config["output_dim"]

    assert linear_layer.input_dim == input_dim, (
        f"Expected input_dim {input_dim}, got {linear_layer.input_dim}"
    )
    assert linear_layer.output_dim == output_dim, (
        f"Expected output_dim {output_dim}, got {linear_layer.output_dim}"
    )

    # Check weights
    assert hasattr(linear_layer, "W"), "Weights W not initialized"
    assert linear_layer.W.shape == (input_dim, output_dim), (
        f"Expected W shape {(input_dim, output_dim)}, got {linear_layer.W.shape}"
    )
    assert np.all(linear_layer.W != 0), "Weights W should not be initialized to zero"

    # Check bias
    assert hasattr(linear_layer, "b"), "Bias b not initialized"
    assert linear_layer.b.shape == (output_dim,), (
        f"Expected b shape {(output_dim,)}, got {linear_layer.b.shape}"
    )
    assert np.all(linear_layer.b == 0), "Bias b should be initialized to zero"


def test_linear_get_parameters(linear_layer: Linear) -> None:
    """Tests get_parameters method."""
    params = linear_layer.get_parameters()

    assert "W" in params, "Weights W not found in parameters"
    assert "b" in params, "Bias b not found in parameters"

    assert np.array_equal(params["W"], linear_layer.W), (
        "Weights W in parameters do not match layer weights"
    )
    assert np.array_equal(params["b"], linear_layer.b), (
        "Bias b in parameters do not match layer bias"
    )

    assert len(params) == 2, "Expected 2 parameters (W and b)"


def test_linear_forward_shape_2d(
    linear_layer: Linear, sample_input_2d: ndarray
) -> None:
    """Tests forward method with 2D input."""
    output = linear_layer(sample_input_2d)
    batch_size = sample_input_2d.shape[0]
    expected_shape = (batch_size, linear_layer.output_dim)

    assert output.shape == expected_shape, (
        f"Expected output shape {expected_shape}, got {output.shape}"
    )


def test_linear_forward_shape_3d(
    linear_layer: Linear, sample_input_3d: ndarray
) -> None:
    """Tests forward method with 3D input."""
    output = linear_layer(sample_input_3d)
    batch_size, seq_len = sample_input_3d.shape[0], sample_input_3d.shape[1]
    expected_shape = (batch_size, seq_len, linear_layer.output_dim)

    assert output.shape == expected_shape, (
        f"Expected output shape {expected_shape}, got {output.shape}"
    )


def test_linear_forward_computation(
    linear_layer: Linear, layer_config: dict[str, int]
) -> None:
    """Tests forward pass computation with set weights and input."""
    input_dim = layer_config["input_dim"]
    output_dim = layer_config["output_dim"]
    batch_size = 2

    # input data
    x = np.array(
        [[1.0 * i for i in range(input_dim)] for _ in range(batch_size)]
    )  # Shape (2, 10)

    # set weights and bias
    W = (
        np.arange(input_dim * output_dim).reshape((input_dim, output_dim)) * 0.1
    )  # Shape (10, 5)
    b = np.arange(output_dim) * 0.1 + 0.1  # Shape (5,)

    linear_layer.set_parameters({"W": W, "b": b})

    # Calculate expected output manually
    expected_output = x @ W + b

    # Calculate actual output using the layer
    actual_output = linear_layer(x)

    # Compare results using np.testing.assert_allclose for float precision
    np.testing.assert_allclose(
        actual_output,
        expected_output,
        rtol=1e-6,
        atol=1e-6,
        err_msg="Forward pass computation mismatch",
    )
