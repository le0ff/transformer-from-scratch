from typing import Any

import numpy as np
import pytest
from numpy import ndarray
from numpy.testing import assert_allclose

from src.layers.dropout import Dropout


@pytest.fixture
def dropout_rate() -> float:
    """Fixture for dropout rate."""
    return 0.5


@pytest.fixture
def dropout_layer(dropout_rate: float) -> Dropout:
    """Fixture for Dropout layer."""
    return Dropout(rate=dropout_rate)


@pytest.fixture
def dropout_layer_seeded(dropout_rate: float) -> Dropout:
    """Fixture for seeded Dropout layer for reproducible masks."""
    return Dropout(rate=dropout_rate, seed=42)


@pytest.fixture
def sample_input_2d() -> ndarray:
    """Sample 2D input data. Using ones simplifies scaling verification."""
    # Shape: (batch_size, features) = (3, 5)
    return np.ones((3, 5), dtype=np.float32)


@pytest.fixture
def sample_input_3d() -> ndarray:
    """Sample 3D input data."""
    # Shape: (batch_size, seq_len, features) = (2, 4, 6)
    return np.ones((2, 4, 6), dtype=np.float32)


# --- Test Functions ---
def test_dropout_init(dropout_layer: Dropout, dropout_rate: float) -> None:
    """Tests initialization of Dropout layer."""
    assert dropout_layer.rate == dropout_rate
    assert dropout_layer.training is True  # Default is training mode
    assert dropout_layer._mask is None  # Mask should be None initially


@pytest.mark.parametrize(
    "invalid_rate",
    [-0.1, 1.0, 1.1, "abc", None],
    ids=["negative", "one", "greater_than_one", "string", "none"],
)
def test_dropout_init_invalid_rate(invalid_rate: Any) -> None:
    """Tests that Dropout layer raises ValueError for invalid rates."""
    with pytest.raises(ValueError):
        Dropout(rate=invalid_rate)


def test_dropout_seed_reproducibility(
    dropout_rate: float, sample_input_2d: ndarray
) -> None:
    """Tests that the dropout mask is reproducible with a fixed seed."""
    seed = 123
    layer1 = Dropout(rate=dropout_rate, seed=seed)
    layer2 = Dropout(rate=dropout_rate, seed=seed)
    # different seed for layer3
    layer3 = Dropout(rate=dropout_rate, seed=seed + 1)

    # train mode by default, but for clarity explicitly set
    layer1.train()
    layer2.train()
    layer3.train()

    output1 = layer1(sample_input_2d)
    mask1 = layer1._mask.copy()
    output2 = layer2(sample_input_2d)
    mask2 = layer2._mask.copy()
    output3 = layer3(sample_input_2d)
    mask3 = layer3._mask.copy()

    assert mask1 is not None and mask2 is not None
    # Same seed should produce same mask
    np.testing.assert_array_equal(
        mask1, mask2, "Masks should be identical for the same seed"
    )
    # Same seed and input should produce same output
    np.testing.assert_array_equal(
        output1, output2, "Outputs should be identical for the same seed"
    )

    # Different seed should produce different mask
    assert not np.array_equal(mask1, mask3), "Masks should differ for different seeds"
    # Different seed and input should produce different output
    assert not np.array_equal(output1, output3), (
        "Outputs should differ for different seeds"
    )


def test_dropout_forward_shape(
    dropout_layer: Dropout, sample_input_2d: ndarray, sample_input_3d: ndarray
) -> None:
    """Tests forward pass shape for 2D and 3D inputs."""
    # --- Training Mode ---
    dropout_layer.train()

    output_2d = dropout_layer(sample_input_2d.copy())
    assert output_2d.shape == sample_input_2d.shape, (
        "2D Output shape should match 2D input shape in training mode."
    )

    output_3d = dropout_layer(sample_input_3d.copy())
    assert output_3d.shape == sample_input_3d.shape, (
        "3D Output shape should match 3D input shape in training mode."
    )

    # --- Evaluation Mode ---
    dropout_layer.eval()

    output_2d_eval = dropout_layer(sample_input_2d.copy())
    assert output_2d_eval.shape == sample_input_2d.shape, (
        "2D Output shape should match 2D input shape in evaluation mode."
    )

    output_3d_eval = dropout_layer(sample_input_3d.copy())
    assert output_3d_eval.shape == sample_input_3d.shape, (
        "3D Output shape should match 3D input shape in evaluation mode."
    )


@pytest.mark.parametrize(
    "sample_input_fixture",
    ["sample_input_2d", "sample_input_3d"],
    ids=["input_2D", "input_3D"],
)
def test_dropout_forward_train(
    dropout_layer_seeded: Dropout,
    dropout_rate: float,
    sample_input_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    """Tests forward pass mask application and scaling in train mode (parametrized)."""
    sample_input = request.getfixturevalue(sample_input_fixture)

    dropout_layer_seeded.train()
    expected_scaling_factor = 1.0 / (1.0 - dropout_rate)
    input_data = sample_input.copy()

    output = dropout_layer_seeded(input_data)
    mask = dropout_layer_seeded._mask

    assert mask is not None, "Mask should be generated in training mode."
    assert mask.shape == input_data.shape, (
        f"Mask shape {mask.shape} should match input shape {input_data.shape}."
    )
    assert np.all(np.isin(mask, [0.0, 1.0])), "Mask should contain only 0.0s and 1.0s."

    assert np.all(output[mask == 0] == 0.0), (
        "Output elements should be 0 where mask is 0."
    )

    expected_scaled_values = input_data[mask == 1] * expected_scaling_factor
    actual_scaled_values = output[mask == 1]
    assert_allclose(
        actual_scaled_values,
        expected_scaled_values,
        err_msg="Output elements should be correctly scaled where mask is 1.",
    )


@pytest.mark.parametrize(
    "sample_input_fixture",
    ["sample_input_2d", "sample_input_3d"],
    ids=["input_2D", "input_3D"],
)
def test_dropout_forward_eval(
    dropout_layer_seeded: Dropout,
    sample_input_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    """Tets forward pass without dropout in eval mode (parameterized)."""
    sample_input = request.getfixturevalue(sample_input_fixture)

    dropout_layer_seeded.eval()

    input_data = sample_input.copy()
    output = dropout_layer_seeded(input_data)
    mask = dropout_layer_seeded._mask

    assert mask is None, "Mask should be None in evaluation mode."
    assert np.array_equal(output, input_data), (
        "Output should be identical to input in eval mode."
    )
