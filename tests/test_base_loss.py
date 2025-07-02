import os
import pytest
import jax.numpy as jnp
import sys

from pathlib import Path

current_dir = Path(os.getcwd())
sys.path.append(str(current_dir))

from src.model_training.base_losses import get_masked_mse

@pytest.mark.parametrize(
    "pred, data, mask, expected",
    [
        # Only the first row is considered (mask = [1.0, 0.0])
        (
            jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            jnp.array([[1.0, 2.0], [0.0, 0.0]]),
            jnp.array([1.0, 0.0]),
            0.0,
        ),
        # Both rows are considered (mask = [1.0, 1.0])
        (
            jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            jnp.array([[0.0, 2.0], [2.0, 4.0]]),
            jnp.array([1.0, 1.0]),
            0.5,
        ),
        # No rows considered (mask = [0.0, 0.0]) – expect NaN or error
        (
            jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            jnp.array([0.0, 0.0]),
            jnp.nan,
        ),
    ],
)
def test_get_masked_mse(pred, data, mask, expected):
    result = get_masked_mse(pred, data, mask)
    if jnp.isnan(expected):
        assert jnp.isnan(result), f"Expected NaN, got {result}"
    else:
        assert jnp.isclose(result, expected), f"Expected {expected}, got {result}"