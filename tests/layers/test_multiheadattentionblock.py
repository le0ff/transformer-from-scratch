import numpy as np
import pytest
from numpy import ndarray

from src.layers.multiheadattentionblock import MultiHeadAttentionBlock

# test output shape , done
# test forward path with known input and output with Dropout disabled , done
# Single-head vs multi-head equivalence
# Mask test:
# Deterministic behavior with fixed seed:
# invalid parameters or dimensions
# test get_parameters and set_parameters methods , done


@pytest.fixture
def layer_config() -> dict[str, int]:
    """Basic layer configuration."""
    return {"d_model": 10, "n_heads": 2, "dropout_rate": 0.0}


@pytest.fixture
def multihead_attention_block(layer_config: dict[str, int]) -> MultiHeadAttentionBlock:
    """Parameterized MultiHeadAttentionBlock fixture."""
    return MultiHeadAttentionBlock(
        d_model=layer_config["d_model"],
        n_heads=layer_config["n_heads"],
        dropout_rate=layer_config["dropout_rate"],
    )


@pytest.fixture
def input_parameters() -> dict[str, int]:
    """Parameters for the input data"""
    return {"batch_size": 3, "seq_len": 8}


@pytest.fixture
def sample_input_3d(
    layer_config: dict[str, int], input_parameters: dict[str, int]
) -> ndarray:
    """Sample 3D input data."""
    # Set a fixed seed for reproducibility
    rng = np.random.default_rng(42)
    return rng.standard_normal(
        (
            input_parameters["batch_size"],
            input_parameters["seq_len"],
            layer_config["d_model"],
        )
    )


@pytest.fixture
def sample_causal_mask(
    layer_config: dict[str, int], input_parameters: dict[str, int]
) -> ndarray:
    """Sample causal mask."""
    # Set a fixed seed for reproducibility
    rng = np.random.default_rng(42)
    mask = np.tril(
        np.ones(
            (input_parameters["seq_len"], input_parameters["seq_len"]), dtype=np.float32
        )
    )
    mask = (
        rng.standard_normal(
            (
                input_parameters["batch_size"],
                layer_config["n_heads"],
                input_parameters["seq_len"],
                input_parameters["seq_len"],
            )
        )
        * mask
    )
    return mask


def test_output_shape(
    multihead_attention_block: MultiHeadAttentionBlock,
    sample_input_3d: ndarray,
    sample_causal_mask: ndarray,
) -> None:
    """
    Test that the output shape is correct.
    """
    output = multihead_attention_block(
        sample_input_3d, sample_input_3d, sample_input_3d, sample_causal_mask
    )
    assert isinstance(output, np.ndarray), "Output is not a numpy array."
    assert output.shape == sample_input_3d.shape, (
        "Output shape does not match input shape."
    )


def test_get_parameters(
    multihead_attention_block: MultiHeadAttentionBlock,
) -> None:
    """
    Test get_parameters method.
    """
    parameters = multihead_attention_block.get_parameters()
    assert isinstance(parameters, dict), "Parameters should be a dictionary."
    assert len(parameters) == 4, "Expected 4 parameters."
    assert "w_q" in parameters, "w_q parameter not found."
    assert "w_k" in parameters, "w_k parameter not found."
    assert "w_v" in parameters, "w_v parameter not found."
    assert "w_o" in parameters, "w_o parameter not found."


def test_set_parameters(
    multihead_attention_block: MultiHeadAttentionBlock,
) -> None:
    """
    Test set_parameters method.
    """
    d_model = multihead_attention_block.d_model

    # Create new valid parameters
    new_w_q = np.random.randn(d_model, d_model) * 5
    new_w_k = np.random.randn(d_model, d_model) * 5
    new_w_v = np.random.randn(d_model, d_model) * 5
    new_w_o = np.random.randn(d_model, d_model) * 5
    params_to_set = {
        "w_q": new_w_q.copy(),
        "w_k": new_w_k.copy(),
        "w_v": new_w_v.copy(),
        "w_o": new_w_o.copy(),
    }

    multihead_attention_block.set_parameters(params_to_set)

    assert np.array_equal(multihead_attention_block.w_q.W, new_w_q)
    assert np.array_equal(multihead_attention_block.w_k.W, new_w_k)
    assert np.array_equal(multihead_attention_block.w_v.W, new_w_v)
    assert np.array_equal(multihead_attention_block.w_o.W, new_w_o)


def test_forward_path() -> None:
    """
    Test the forward path.
    """
    mhab = MultiHeadAttentionBlock(d_model=4, n_heads=1, dropout_rate=0.0)
    x = np.array([[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]])
    mask = np.array([[[[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]]])
    d_model = mhab.d_model
    weights = np.ones((d_model, d_model), dtype=np.float32)
    params_to_set = {
        "w_q": weights.copy(),
        "w_k": weights.copy(),
        "w_v": weights.copy(),
        "w_o": weights.copy(),
    }
    mhab.set_parameters(params_to_set)
    output = mhab(x, x, x, mask)
    print(output)
    expected_output = np.array(
        [
            [
                [40, 40, 40, 40],
                [40, 40, 40, 40],
                [40, 40, 40, 40],
                [40, 40, 40, 40],
            ]
        ]
    )
    assert np.array_equal(output, expected_output), (
        "Output does not match expected output."
    )
