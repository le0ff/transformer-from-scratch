import re
from typing import Dict

import numpy as np
import pytest
from numpy import ndarray
from numpy.testing import assert_array_equal

from src.layers.projection import ProjectionLayer

# --- Fixtures ---


@pytest.fixture
def proj_config() -> Dict[str, int]:
    """Basic projection layer configuration."""
    return {"d_model": 8, "vocab_size": 20}


@pytest.fixture
def proj_seed() -> int:
    """Seed for the projection layer."""
    return 123


@pytest.fixture
def proj_layer(proj_config: Dict[str, int], proj_seed: int) -> ProjectionLayer:
    """Fixture for creating a ProjectionLayer instance."""
    return ProjectionLayer(
        d_model=proj_config["d_model"],
        vocab_size=proj_config["vocab_size"],
        seed=proj_seed,
    )


@pytest.fixture
def sample_input_2d(proj_config: Dict[str, int]) -> ndarray:
    """Sample 2D input for the projection layer."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((4, proj_config["d_model"]))  # .astype(np.float32)


@pytest.fixture
def sample_input_3d(proj_config: Dict[str, int]) -> ndarray:
    """Sample 3D input for the projection layer."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((4, 5, proj_config["d_model"]))


# --- Tests ---


@pytest.mark.parametrize(
    "input_fixture",
    ["sample_input_2d", "sample_input_3d"],
    ids=["2D_Input", "3D_Input"],
)
def test_projection_layer_forward(
    proj_layer: ProjectionLayer,
    request: pytest.FixtureRequest,
    input_fixture: str,
) -> None:
    """Test the forward pass of the projection layer."""
    input_data = request.getfixturevalue(input_fixture)
    output = proj_layer.forward(input_data)

    # Check output shape
    expected_shape = input_data.shape[:-1] + (proj_layer.vocab_size,)
    assert output.shape == expected_shape, (
        f"Expected output shape {expected_shape}, got {output.shape}"
    )

    # Check output values (log-probabilities)
    assert np.all(np.isfinite(output)), "Output contains non-finite values."
    assert np.all(np.isclose(np.sum(np.exp(output), axis=-1), 1)), (
        "Output does not sum to 1 along the last dimension."
    )


def test_projection_get_parameters(proj_layer: ProjectionLayer) -> None:
    params = proj_layer.get_parameters()
    assert "linear_W" in params
    assert "linear_b" in params
    assert params["linear_W"].shape == (
        proj_layer.d_model,
        proj_layer.vocab_size,
    )
    assert params["linear_b"].shape == (proj_layer.vocab_size,)


def test_projection_set_parameters_valid(proj_layer: ProjectionLayer) -> None:
    d_model = proj_layer.d_model
    vocab_size = proj_layer.vocab_size
    new_W = np.random.randn(d_model, vocab_size)
    new_b = np.random.randn(vocab_size)
    params = {"linear_W": new_W, "linear_b": new_b}
    proj_layer.set_parameters(params)
    assert_array_equal(proj_layer.linear.W, new_W)
    assert_array_equal(proj_layer.linear.b, new_b)


@pytest.mark.parametrize(
    "params, error_type, error_msg",
    [
        ({}, ValueError, "No parameters provided"),
        ({"wrongprefix_W": np.zeros((2, 2))}, ValueError, "Unexpected parameter key"),
        (
            {"linear_W": np.zeros((2, 2))},
            ValueError,
            re.escape("Expected W shape (8, 20), but got (2, 2)"),
        ),
        (
            {"linear_W": np.zeros((8, 20))},
            ValueError,
            "Bias parameter 'b' missing in params dictionary, but use_bias is True.",
        ),
        (
            {"linear_b": np.zeros((2, 2))},
            ValueError,
            "Weight parameter 'W' missing in params dictionary.",
        ),
    ],
)
def test_projection_set_parameters_invalid(
    proj_layer: ProjectionLayer, params, error_type, error_msg
):
    with pytest.raises(error_type, match=error_msg):
        proj_layer.set_parameters(params)


def test_projection_train_eval_modes(proj_layer: ProjectionLayer) -> None:
    proj_layer.eval()
    assert not proj_layer.training
    assert not proj_layer.linear.training
    proj_layer.train()
    assert proj_layer.training
    assert proj_layer.linear.training


@pytest.mark.parametrize(
    "input_fixture",
    ["sample_input_2d", "sample_input_3d"],
    ids=["2D_Input", "3D_Input"],
)
def test_projection_logsoftmax_equivalence(
    proj_layer: ProjectionLayer,
    request: pytest.FixtureRequest,
    input_fixture: str,
) -> None:
    """Verify forward() matches manual numerically stable log-softmax."""
    input_data = request.getfixturevalue(input_fixture)
    # Output from the layer
    out_logsoftmax = proj_layer.forward(input_data)

    # Manual log-softmax, as implemented before adjusting forward of projection layer
    logits = proj_layer.linear(input_data)
    max_logits = np.max(logits, axis=-1, keepdims=True)
    shifted = logits - max_logits
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    manual_log_probs = shifted - log_sum_exp
    # Compare
    np.testing.assert_allclose(
        out_logsoftmax,
        manual_log_probs,
        atol=1e-6,
        rtol=1e-6,
        err_msg=f"forward() and manual log-softmax differ for {input_fixture}",
    )
