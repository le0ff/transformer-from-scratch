import numpy as np
import pytest

from src.layers.activations.softmax import Softmax


@pytest.fixture
def small_input() -> np.ndarray:
    return np.array([2.0, 1.0, 1.0])


@pytest.fixture
def big_input() -> np.ndarray:
    return np.array([1002.0, 1001.0, 1001.0])


@pytest.fixture
def small_input_2d() -> np.ndarray:
    return np.array([[2.0, 1.0, 1.0], [2.0, 1.0, 1.0], [2.0, 1.0, 1.0]])


@pytest.fixture
def small_input_2d_mask() -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])


@pytest.fixture
def softmax_activation() -> Softmax:
    return Softmax(axis=-1)


def test_sum_to_one(softmax_activation: Softmax, small_input: np.ndarray) -> None:
    """
    Test that the softmax output sums to 1.
    """
    output = softmax_activation(small_input)
    assert isinstance(output, np.ndarray), "Output is not a numpy array."
    assert np.isclose(np.sum(output), 1.0), "Softmax output does not sum to 1."


def test_shape(softmax_activation: Softmax, small_input: np.ndarray) -> None:
    """
    Test that the softmax output has the same shape as the input.
    """
    output = softmax_activation(small_input)
    assert output.shape == small_input.shape, (
        "Softmax output shape does not match input shape."
    )


def test_numerical_stability(
    softmax_activation: Softmax, small_input: np.ndarray, big_input: np.ndarray
) -> None:
    """
    Test that the softmax function is numerically stable.
    """
    output_small = softmax_activation(small_input)
    output_big = softmax_activation(big_input)

    # Check that the outputs are close to each other
    assert np.allclose(output_small, output_big), (
        "Softmax outputs are not numerically stable."
    )


def test_2d_input(softmax_activation: Softmax, small_input_2d: np.ndarray) -> None:
    """
    Test that the softmax function works with 2D input.
    """
    output = softmax_activation(small_input_2d)
    print(output, output.shape)
    assert isinstance(output, np.ndarray), "Output is not a numpy array."
    assert output.shape == small_input_2d.shape, (
        "Softmax output shape does not match input shape."
    )
    assert np.allclose(np.sum(output, axis=-1), 1.0), (
        "Softmax output does not sum to 1 along the last axis."
    )


def test_causal_mask(
    softmax_activation: Softmax,
    small_input_2d: np.ndarray,
    small_input_2d_mask: np.ndarray,
) -> None:
    """
    Test that the softmax function works with a causal mask.
    """
    output = softmax_activation(x=small_input_2d, causal_mask=small_input_2d_mask)
    assert isinstance(output, np.ndarray), "Output is not a numpy array."
    assert output.shape == small_input_2d.shape, (
        "Softmax output shape does not match input shape."
    )
    assert np.allclose(np.sum(output, axis=-1), 1.0), (
        "Softmax output does not sum to 1 along the last axis."
    )
    assert np.allclose(
        output,
        [[1.0, 0.0, 0.0], [0.731059, 0.268941, 0.0], [0.576117, 0.211942, 0.211942]],
    )
