import equinox as eqx
import jax.numpy as jnp

from jaxtyping import Array
from src.model_training.base_losses import get_masked_mse


def regularise_mlp(
        mlp: eqx.nn.MLP,
        reg_exp: float,
        reg_strength: float,
):
    """
    Regularise the weights of an MLP with L1/L2 regularisation depending on `reg_exp`.
    """
    reg_loss = 0
    for layer in mlp.layers:
        if isinstance(layer, eqx.nn.Linear):
            reg_loss += jnp.mean(jnp.abs(layer.weight) ** reg_exp)
            if hasattr(layer, 'bias') and layer.bias is not None:
                reg_loss += jnp.mean(jnp.abs(layer.bias) ** reg_exp)
    return reg_strength * reg_loss


def regularise_mlp_contribution(
        l2_norm_nn_contrib: jnp.ndarray,
        timespan_length: float,
        reg_strength: float,
):
    """
    Regularise neural network contribution to UDE as per non-negative UDE paper, section 3.2.
    ref: https://arxiv.org/pdf/2406.14246
    """
    # l2_norm_nn_contrib[-1] is the integral up to the final timepoint.
    # 1e-6 pre-multiplier for reg_strength avoids wildcard issues in snakemake
    if timespan_length > 0:
        return 1e-6 * reg_strength * (l2_norm_nn_contrib[-1]/timespan_length)**2
    else:
        # Handles cases where the last window contains a single datapoint
        return 0.0


def compute_discontinuity_penalty(
        solutions: Array,
        state_dimension: int,
        reg_strength: float,
        masks: Array,
):
    """
    Compute the discontinuity penalty by measuring the sum of square errors
    between the last state of each shooting window and the first state of the next.
    """
    if solutions.shape[0] > 1:  # valid for MS or for MS+CL before last stage
        discontinuity_mse = get_masked_mse(
            solutions[:-1, :, :state_dimension],  # endpoint
            solutions[1:, :, :state_dimension],  # startpoint
            masks[:-1].astype(int) & masks[1:].astype(int)
        )
    else:  # for MS+CL final CL stage (single valid window)
        discontinuity_mse = 0
    return reg_strength * discontinuity_mse
