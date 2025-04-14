from typing import Any

import numpy as np
import pytest
from numpy import ndarray

from src.layers.linear import Linear


@pytest.fixture
def layer_config() -> dict[str, int]:
    """Basic layer configuration."""
    return {"input_dim": 10, "output_dim": 5}


@pytest.fixture(params=[True, False], ids=["with_bias", "without_bias"])
def use_bias(request: pytest.FixtureRequest) -> bool:
    """Fixture for use_bias parameter."""
    return request.param


@pytest.fixture
def linear_layer(layer_config: dict[str, int], use_bias: bool) -> Linear:
    """Parameterized Linear Layer fixture (with and without bias)."""
    return Linear(
        input_dim=layer_config["input_dim"],
        output_dim=layer_config["output_dim"],
        use_bias=use_bias,
        seed=42,
    )


@pytest.fixture
def sample_input_2d(layer_config: dict[str, int]) -> ndarray:
    """Sample 2D input data."""
    batch_size = 4
    # Set a fixed seed for reproducibility
    rng = np.random.default_rng(123)
    return rng.standard_normal((batch_size, layer_config["input_dim"]))


@pytest.fixture
def sample_input_3d(layer_config: dict[str, int]) -> ndarray:
    """Sample 3D input data."""
    batch_size = 4
    seq_len = 8
    # Set a fixed seed for reproducibility
    rng = np.random.default_rng(123)
    return rng.standard_normal((batch_size, seq_len, layer_config["input_dim"]))


# --- Test Functions ---
def test_linear_init(linear_layer: Linear, layer_config: dict[str, int]) -> None:
    """Tests initialization of linear layer: shape, dimensions ."""
    input_dim = layer_config["input_dim"]
    output_dim = layer_config["output_dim"]

    assert linear_layer.input_dim == input_dim, (
        f"Expected input_dim {input_dim}, got {linear_layer.input_dim}"
    )
    assert linear_layer.output_dim == output_dim, (
        f"Expected output_dim {output_dim}, got {linear_layer.output_dim}"
    )

    # Check weights
    assert hasattr(linear_layer, "W"), "Weights W does not exist"
    assert linear_layer.W.shape == (input_dim, output_dim), (
        f"Expected W shape {(input_dim, output_dim)}, got {linear_layer.W.shape}"
    )
    assert np.all(linear_layer.W != 0), "Weights W should not be initialized to zero"

    # Check bias
    if linear_layer.use_bias:
        assert hasattr(linear_layer, "b"), "Bias b does not exist"
        assert linear_layer.b.shape == (output_dim,), (
            f"Expected b shape {(output_dim,)}, got {linear_layer.b.shape}"
        )
        assert np.all(linear_layer.b == 0), "Bias b should be initialized to zero"
    else:
        assert hasattr(linear_layer, "b"), (
            "Attribute b should exist even if use_bias=False (set to None)"
        )
        assert linear_layer.b is None, "Bias b should be None when use_bias=False"


@pytest.mark.parametrize(
    "invalid_config, expected_error",
    [
        ({"input_dim": 0, "output_dim": 5}, ValueError),
        ({"input_dim": 10, "output_dim": 0}, ValueError),
        ({"input_dim": -1, "output_dim": 5}, ValueError),
        ({"input_dim": 10, "output_dim": -5}, ValueError),
        ({"input_dim": 10, "output_dim": 5, "seed": "abc"}, ValueError),
        ({"input_dim": 10, "output_dim": 5, "seed": 1.5}, ValueError),
    ],
    ids=[
        "zero_input_dim",
        "zero_output_dim",
        "neg_input_dim",
        "neg_output_dim",
        "bad_seed_type_str",
        "bad_seed_type_float",
    ],
)
def test_linear_init_errors(
    invalid_config: dict[str, Any], expected_error: type[Exception]
) -> None:
    """Tests that Linear layer raises errors on invalid initialization parameters."""
    input_dim = invalid_config.get("input_dim", 10)
    output_dim = invalid_config.get("output_dim", 5)
    seed = invalid_config.get("seed", None)

    with pytest.raises(expected_error):
        Linear(input_dim=input_dim, output_dim=output_dim, seed=seed)


def test_linear_get_parameters(linear_layer: Linear) -> None:
    """Tests get_parameters method."""
    params = linear_layer.get_parameters()

    assert "W" in params, "Weights W not found in parameters"
    assert np.array_equal(params["W"], linear_layer.W), (
        "Weights W in parameters do not match layer weights"
    )

    if linear_layer.use_bias:
        assert "b" in params, "Bias b not found in parameters"
        assert np.array_equal(params["b"], linear_layer.b), (
            "Bias b in parameters do not match layer bias"
        )
        assert len(params) == 2, "Expected 2 parameters (W and b)"
    else:
        assert "b" not in params, "Bias b should not be present in parameters"
        assert len(params) == 1, "Expected 1 parameters (W)"


def test_linear_set_parameters_valid(linear_layer: Linear) -> None:
    """Tests setting valid parameters, considering use_bias."""
    input_dim = linear_layer.input_dim
    output_dim = linear_layer.output_dim

    # Create new valid parameters
    new_W = np.random.randn(input_dim, output_dim) * 5
    params_to_set = {"W": new_W.copy()}

    if linear_layer.use_bias:
        new_b = np.random.randn(output_dim) * 2
        params_to_set["b"] = new_b.copy()

    # Set parameters
    linear_layer.set_parameters(params_to_set)

    # Verify they were set correctly
    assert np.array_equal(linear_layer.W, new_W)
    if linear_layer.use_bias:
        assert linear_layer.b is not None
        assert np.array_equal(linear_layer.b, new_b)
    else:
        assert linear_layer.b is None


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

    # prepare parameters to set and calculate expected output
    params_to_set = {"W": W}
    expected_output = x @ W
    if linear_layer.use_bias:
        params_to_set["b"] = b
        expected_output = expected_output + b

    # Set parameters
    linear_layer.set_parameters(params_to_set)

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


def test_linears_same_seed(layer_config: dict[str, int], use_bias: bool) -> None:
    """Tests initialization of two linear layers with same seed."""
    input_dim = layer_config["input_dim"]
    output_dim = layer_config["output_dim"]
    seed = 42

    layer1 = Linear(input_dim, output_dim, use_bias=use_bias, seed=seed)
    layer2 = Linear(input_dim, output_dim, use_bias=use_bias, seed=seed)

    # Check if weights are initialized equally
    assert np.array_equal(layer1.W, layer2.W), "Weights W should be equal"
    if use_bias:
        assert np.array_equal(layer1.b, layer2.b), (
            "Bias b should be equal, as it is initialized to zero"
        )
    else:
        assert layer1.b is None and layer2.b is None, "Bias should be None for both"


def test_linears_diff_seed(layer_config: dict[str, int], use_bias: bool) -> None:
    """Tests initialization of two linear layers with different seed."""
    input_dim = layer_config["input_dim"]
    output_dim = layer_config["output_dim"]
    seed = 42

    layer1 = Linear(input_dim, output_dim, use_bias=use_bias, seed=seed)
    layer2 = Linear(input_dim, output_dim, use_bias=use_bias, seed=seed + 1)

    # Check if weights are initialized differently
    assert not np.array_equal(layer1.W, layer2.W), "Weights W should be different"
    if use_bias:
        assert np.array_equal(layer1.b, layer2.b), (
            "Bias b should be equal, as it is initialized to zero"
        )
    else:
        assert layer1.b is None and layer2.b is None, "Bias should be None for both"


def test_linears_no_seed(layer_config: dict[str, int], use_bias: bool) -> None:
    """Tests initialization of two linear layers with no seed (should be different)."""
    input_dim = layer_config["input_dim"]
    output_dim = layer_config["output_dim"]

    layer1 = Linear(input_dim, output_dim, use_bias=use_bias)
    layer2 = Linear(input_dim, output_dim, use_bias=use_bias)

    # Check if weights and biases are initialized differently
    assert not np.array_equal(layer1.W, layer2.W), (
        "Weights W should (likely) be different for no seed"
    )
    if use_bias:
        assert np.array_equal(layer1.b, layer2.b), (
            "Bias b should be equal, as it is initialized to zero"
        )
    else:
        assert layer1.b is None and layer2.b is None, "Bias should be None for both"
