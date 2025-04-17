import re
from typing import Dict

import numpy as np
import pytest
from numpy import ndarray
from numpy.testing import assert_allclose, assert_array_equal

from src.layers.feedforward import FeedForwardBlock


@pytest.fixture
def ff_config() -> Dict:
    """Basic FeedForwardBlock configuration."""
    return {"d_model": 16, "d_ff": 32, "dropout": 0.1}


@pytest.fixture
def ff_seed() -> int:
    """Seed for reproducible tests."""
    return 42


@pytest.fixture
def ff_block(ff_config: dict) -> FeedForwardBlock:
    """Fixture for FeedForwardBlock."""
    return FeedForwardBlock(
        d_model=ff_config["d_model"],
        d_ff=ff_config["d_ff"],
        dropout=ff_config["dropout"],
        seed=None,
    )


@pytest.fixture
def ff_block_seeded(ff_config: dict, ff_seed: int) -> FeedForwardBlock:
    """Fixture for FeedForwardBlock with a seed."""
    return FeedForwardBlock(
        d_model=ff_config["d_model"],
        d_ff=ff_config["d_ff"],
        dropout=ff_config["dropout"],
        seed=ff_seed,
    )


@pytest.fixture
def sample_input_ff_2d(ff_config: dict) -> ndarray:
    """Sample 2D input data (batch, d_model)."""
    batch_size = 4
    rng = np.random.default_rng(123)
    return rng.standard_normal((batch_size, ff_config["d_model"])).astype(np.float32)


@pytest.fixture
def sample_input_ff_3d(ff_config: dict) -> ndarray:
    """Sample 3D input data (batch, seq_len, d_model)."""
    batch_size = 4
    seq_len = 8
    rng = np.random.default_rng(123)
    return rng.standard_normal((batch_size, seq_len, ff_config["d_model"])).astype(
        np.float32
    )


# --- Test Functions ---+


@pytest.mark.parametrize(
    "invalid_params, error_msg_match",
    [
        (
            {"d_model": 0, "d_ff": 32, "dropout": 0.1},
            "d_model and d_ff must be positive",
        ),
        (
            {"d_model": 16, "d_ff": 0, "dropout": 0.1},
            "d_model and d_ff must be positive",
        ),
        (
            {"d_model": 16, "d_ff": -5, "dropout": 0.1},
            "d_model and d_ff must be positive",
        ),
        (
            {"d_model": 16, "d_ff": 32, "dropout": -0.1},
            re.escape("Dropout must be a float in [0.0, 1.0)"),
        ),
        (
            {"d_model": 16, "d_ff": 32, "dropout": 1.0},
            re.escape("Dropout must be a float in [0.0, 1.0)"),
        ),
        (
            {"d_model": 16, "d_ff": 32, "dropout": 1.1},
            re.escape("Dropout must be a float in [0.0, 1.0)"),
        ),
        (
            {"d_model": 16, "d_ff": 32, "dropout": "abc"},
            re.escape("Dropout must be a float in [0.0, 1.0)"),
        ),
        (
            {"d_model": 16, "d_ff": 32, "dropout": 0.1, "seed": "xyz"},
            "Seed must be an integer",
        ),
        (
            {"d_model": 16, "d_ff": 32, "dropout": 0.1, "seed": 1.5},
            "Seed must be an integer",
        ),
    ],
    ids=[
        "zero_d_model",
        "zero_d_ff",
        "neg_d_ff",
        "neg_dropout",
        "dropout_eq_1",
        "dropout_gt_1",
        "dropout_str",
        "seed_str",
        "seed_float",
    ],
)
def test_ff_init_invalid_params(invalid_params: dict, error_msg_match: str) -> None:
    """Tests that FeedForwardBlock raises ValueError for invalid initialization parameters."""
    with pytest.raises(ValueError, match=error_msg_match):
        FeedForwardBlock(**invalid_params)


def test_ff_train_eval_propagation(ff_block: FeedForwardBlock) -> None:
    """Tests if train/eval mode propagates correctly to sub-layers."""
    # Initial state (should be training)
    assert ff_block.training is True
    assert ff_block.linear1.training is True
    assert ff_block.relu.training is True
    assert ff_block.dropout_layer.training is True
    assert ff_block.linear2.training is True

    # Switch to eval mode
    ff_block.eval()
    assert ff_block.training is False
    assert ff_block.linear1.training is False
    assert ff_block.relu.training is False
    assert ff_block.dropout_layer.training is False
    assert ff_block.linear2.training is False

    # Switch back to train mode
    ff_block.train()
    assert ff_block.training is True
    assert ff_block.linear1.training is True
    assert ff_block.relu.training is True
    assert ff_block.dropout_layer.training is True
    assert ff_block.linear2.training is True


@pytest.mark.parametrize(
    "input_fixture",
    ["sample_input_ff_2d", "sample_input_ff_3d"],
    ids=["2D_Input", "3D_Input"],
)
def test_ff_forward_shape(
    ff_block_seeded: FeedForwardBlock,
    input_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    """Tests the output shape of the forward pass for 2D and 3D inputs."""
    sample_input = request.getfixturevalue(input_fixture)
    d_model = ff_block_seeded.d_model
    expected_shape = list(sample_input.shape)
    expected_shape[-1] = d_model  # Output dimension should match d_model

    # Test in training mode
    ff_block_seeded.train()
    output_train = ff_block_seeded(sample_input.copy())
    assert output_train.shape == tuple(expected_shape), (
        f"Train mode output shape mismatch. Expected {tuple(expected_shape)}, got {output_train.shape}"
    )

    # Test in eval mode
    ff_block_seeded.eval()
    output_eval = ff_block_seeded(sample_input.copy())
    assert output_eval.shape == tuple(expected_shape), (
        f"Eval mode output shape mismatch. Expected {tuple(expected_shape)}, got {output_eval.shape}"
    )


def test_ff_get_parameters(ff_block_seeded: FeedForwardBlock) -> None:
    """Tests the get_parameters method."""
    params = ff_block_seeded.get_parameters()

    expected_keys = {"linear1_W", "linear1_b", "linear2_W", "linear2_b"}
    assert set(params.keys()) == expected_keys, "Parameter keys mismatch."

    # Check shapes and values against actual layer parameters
    assert_array_equal(params["linear1_W"], ff_block_seeded.linear1.W)
    assert params["linear1_W"].shape == (ff_block_seeded.d_model, ff_block_seeded.d_ff)

    assert_array_equal(params["linear1_b"], ff_block_seeded.linear1.b)
    assert params["linear1_b"].shape == (ff_block_seeded.d_ff,)

    assert_array_equal(params["linear2_W"], ff_block_seeded.linear2.W)
    assert params["linear2_W"].shape == (ff_block_seeded.d_ff, ff_block_seeded.d_model)

    assert_array_equal(params["linear2_b"], ff_block_seeded.linear2.b)
    assert params["linear2_b"].shape == (ff_block_seeded.d_model,)


def test_ff_set_parameters_valid(ff_block: FeedForwardBlock) -> None:
    """Tests setting valid parameters using set_parameters."""
    d_model = ff_block.d_model
    d_ff = ff_block.d_ff

    # Create new parameters with different values
    new_params = {
        "linear1_W": np.random.randn(d_model, d_ff).astype(np.float32),
        "linear1_b": np.random.randn(d_ff).astype(np.float32),
        "linear2_W": np.random.randn(d_ff, d_model).astype(np.float32),
        "linear2_b": np.random.randn(d_model).astype(np.float32),
    }

    # Set the new parameters
    ff_block.set_parameters(new_params)

    # Verify the parameters were updated in the layers
    assert_array_equal(ff_block.linear1.W, new_params["linear1_W"])
    assert_array_equal(ff_block.linear1.b, new_params["linear1_b"])
    assert_array_equal(ff_block.linear2.W, new_params["linear2_W"])
    assert_array_equal(ff_block.linear2.b, new_params["linear2_b"])

    # Verify get_parameters reflects the changes
    retrieved_params = ff_block.get_parameters()
    assert len(retrieved_params) == 4
    assert_array_equal(retrieved_params["linear1_W"], new_params["linear1_W"])
    assert_array_equal(retrieved_params["linear2_b"], new_params["linear2_b"])


@pytest.mark.parametrize(
    "invalid_params, error_type, error_match",
    [
        # Wrong shape
        (
            {"linear1_W": np.zeros((1, 1))},
            ValueError,
            "Error setting parameters for sub-layer 'linear1'",
        ),
        # Missing linear2_b
        (
            {
                "linear1_W": np.zeros((16, 32)),
                "linear1_b": np.zeros((32)),
                "linear2_W": np.zeros((32, 16)),
            },
            ValueError,
            "Error setting parameters for sub-layer 'linear2'",
        ),
        # Extra key check
        (
            {
                "linear1_W": np.zeros((16, 32)),
                "linear1_b": np.zeros((32)),
                "linear2_W": np.zeros((32, 16)),
                "linear2_b": np.zeros((16)),
                "extra_param": np.zeros(1),
            },
            ValueError,
            "Missing parameters for layers: {'extra_param'}",
        ),
        # Wrong prefix
        (
            {"wrongprefix_W": np.zeros((16, 32))},
            ValueError,
            "Missing parameters for layers: {'wrongprefix_W'}",
        ),
    ],
    ids=["wrong_shape_l1w", "missing_l2b", "extra_key", "wrong_prefix"],
)
def test_ff_set_parameters_invalid(
    ff_block_seeded: FeedForwardBlock,
    invalid_params: dict,
    error_type: type[Exception],
    error_match: str,
) -> None:
    """Tests that set_parameters raises errors for invalid inputs."""

    with pytest.raises(error_type, match=error_match):
        ff_block_seeded.set_parameters(invalid_params)


def test_ff_seed_derivation_reproducibility(ff_config: dict, ff_seed: int) -> None:
    """Tests that the same seed produces identical blocks, and derived seeds differ."""
    # Create two blocks with the same seed
    block1 = FeedForwardBlock(seed=ff_seed, **ff_config)
    block2 = FeedForwardBlock(seed=ff_seed, **ff_config)

    # Get parameters
    params1 = block1.get_parameters()
    params2 = block2.get_parameters()

    # Check parameters are identical between blocks with same seed
    assert_array_equal(params1["linear1_W"], params2["linear1_W"])
    assert_array_equal(params1["linear1_b"], params2["linear1_b"])
    assert_array_equal(params1["linear2_W"], params2["linear2_W"])
    assert_array_equal(params1["linear2_b"], params2["linear2_b"])

    # Check linear layers are different (within the same block)
    assert not np.array_equal(params1["linear1_W"], params1["linear2_W"]), (
        "Weights of linear1 and linear2 should differ even with the same main seed."
    )

    # Verify dropout reproducibility (by checking forward pass)
    block1.train()
    block2.train()
    input_data = np.random.randn(2, ff_config["d_model"]).astype(np.float32)
    output1 = block1(input_data.copy())
    mask1 = block1.dropout_layer._mask.copy()
    output2 = block2(input_data.copy())
    mask2 = block2.dropout_layer._mask.copy()

    assert_array_equal(
        mask1, mask2, "Dropout masks should be identical for the same seed."
    )
    assert_allclose(
        output1, output2, err_msg="Outputs should be identical for the same seed."
    )


def test_ff_different_seeds(ff_config: dict, ff_seed: int) -> None:
    """Tests that different seeds produce different blocks."""
    block1 = FeedForwardBlock(seed=ff_seed, **ff_config)
    block2 = FeedForwardBlock(seed=ff_seed + 1, **ff_config)

    params1 = block1.get_parameters()
    params2 = block2.get_parameters()

    # Check parameters are different between blocks
    assert not np.array_equal(params1["linear1_W"], params2["linear1_W"])
    assert not np.array_equal(params1["linear2_W"], params2["linear2_W"])

    # Verify dropout difference (by checking forward pass)
    block1.train()
    block2.train()
    input_data = np.random.randn(2, ff_config["d_model"]).astype(np.float32)
    output1 = block1(input_data.copy())
    output2 = block2(input_data.copy())

    assert not np.allclose(output1, output2), (
        "Outputs should differ for different seeds."
    )
