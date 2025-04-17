# from typing import Any

import numpy as np
import pytest

from src.layers.linear import Linear

# from numpy import ndarray
from src.layers.residual import ResidualConnection


# --- Fixtures ---
@pytest.fixture
def layer_config() -> dict[str, int]:
    """Basic layer configuration."""
    return {"normalized_shape": 10, "dropout_rate": 0.0, "eps": 0.75}


@pytest.fixture
def linear_layer(layer_config: dict[str, int]) -> Linear:
    """Linear layer fixture."""
    return Linear(
        input_dim=layer_config["normalized_shape"],
        output_dim=layer_config["normalized_shape"],
        use_bias=False,
        seed=42,
    )


@pytest.fixture
def residual_layer(layer_config: dict[str, int]) -> ResidualConnection:
    """Residual layer fixture."""
    return ResidualConnection(
        normalized_shape=layer_config["normalized_shape"],
        dropout_rate=layer_config["dropout_rate"],
        eps=layer_config["eps"],
    )


# --- Test Functions ---
def test_residual_init(
    residual_layer: ResidualConnection, layer_config: dict[str, int]
) -> None:
    """Tests initialization of residual layer: shape, dimensions."""
    normalized_shape = layer_config["normalized_shape"]
    dropout_rate = layer_config["dropout_rate"]

    assert residual_layer.layer_norm.normalized_shape == normalized_shape, (
        f"Expected normalized_shape {normalized_shape}, got {residual_layer.layer_norm.normalized_shape}"
    )
    assert residual_layer.dropout.rate == dropout_rate, (
        f"Expected dropout_rate {dropout_rate}, got {residual_layer.dropout.dropout_rate}"
    )
    assert residual_layer.layer_norm.training is True, (
        f"Expected training mode to be True, got {residual_layer.layer_norm.training}"
    )


def test_residual_forward(
    residual_layer: ResidualConnection,
    linear_layer: Linear,
    layer_config: dict[str, int],
) -> None:
    """Tests forward pass of residual layer."""
    x = np.random.rand(5, layer_config["normalized_shape"]).astype(
        np.float32
    )  # Example input
    output = residual_layer.forward(x, linear_layer)

    assert output.shape == x.shape, (
        f"Expected output shape {x.shape}, got {output.shape}"
    )


def test_residual_forward_with_invalid_shape(
    residual_layer: ResidualConnection,
    linear_layer: Linear,
) -> None:
    """Tests forward pass of residual layer with invalid input shape."""
    x = np.random.rand(5, 20).astype(np.float32)  # Example input with invalid shape

    with pytest.raises(ValueError):
        residual_layer.forward(x, linear_layer)


def test_residual_forward_with_invalid_sublayer(
    residual_layer: ResidualConnection,
    layer_config: dict[str, int],
) -> None:
    """Tests forward pass of residual layer with invalid sublayer."""
    x = np.random.rand(5, layer_config["normalized_shape"]).astype(
        np.float32
    )  # Example input
    invalid_sublayer = x

    with pytest.raises(TypeError):
        residual_layer.forward(x, invalid_sublayer)


def test_residual_forward_deterministic(
    residual_layer: ResidualConnection,
    linear_layer: Linear,
    layer_config: dict[str, int],
) -> None:
    """Tests forward pass of residual layer for deterministic output."""
    x = np.array(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]
    ).astype(np.float32)
    linear_layer.set_parameters(
        {
            "W": np.eye(
                layer_config["normalized_shape"],
                dtype=np.float32,
            )
        }
    )
    output = residual_layer.forward(x, linear_layer)
    expected_output = np.array(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]
    ).astype(np.float32) + np.array(
        [
            [
                -4.5 / 3,
                -3.5 / 3,
                -2.5 / 3,
                -1.5 / 3,
                -0.5 / 3,
                0.5 / 3,
                1.5 / 3,
                2.5 / 3,
                3.5 / 3,
                4.5 / 3,
            ],
            [
                -4.5 / 3,
                -3.5 / 3,
                -2.5 / 3,
                -1.5 / 3,
                -0.5 / 3,
                0.5 / 3,
                1.5 / 3,
                2.5 / 3,
                3.5 / 3,
                4.5 / 3,
            ],
        ]
    )
    print(f" output: {output}")
    print(f"expected_output: {expected_output}")

    assert np.allclose(output, expected_output), (
        f"Expected output {expected_output}, got {output}"
    )


def test_residual_forward_with_dropout(
    residual_layer: ResidualConnection,
    linear_layer: Linear,
    layer_config: dict[str, int],
) -> None:
    """Tests forward pass of residual layer with dropout."""
    x = np.random.rand(5, layer_config["normalized_shape"]).astype(
        np.float32
    )  # Example input
    residual_layer.dropout.rate = 0.5  # Set dropout rate to 50%
    output = residual_layer.forward(x, linear_layer)

    assert output.shape == x.shape, (
        f"Expected output shape {x.shape}, got {output.shape}"
    )
