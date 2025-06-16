import equinox as eqx
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr
import jax.tree_util as jtu
import numpy as np

from jaxtyping import Array
from typing import Any, Union


def get_mse(pred: Array, data: Array):
    return jnp.mean((pred - data) ** 2)

def get_masked_mse(pred: Array, data: Array, mask: Array):
    # Masked mean square error - mean is computed only over non-zero terms
    return jnp.sum(jnp.expand_dims(mask, axis=-1) * (pred - data) ** 2) / (jnp.sum(mask) * pred.shape[-1])

def get_masked_sse(pred: Array, data: Array, mask: Array):
    # Masked sum square error - sum is computed only over non-zero terms
    return jnp.sum(jnp.expand_dims(mask, axis=-1) * (pred - data) ** 2)


# TODO @GiacomoFabrini - add unit test
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


# TODO @GiacomoFabrini - add integration test
def regularise_mlp_contribution(
        l2_norm_nn_contrib: jnp.ndarray,
        timespan_length: float,
        reg_strength: float,
):
    """
    Regularise neural network contribution to UDE as per non-negative UDE paper, section 3.2.
    ref: https://arxiv.org/pdf/2406.14246
    """
    # the l2_norm_nn_contrib is the integral of the 2-norm of the nn_contribution from t0 to t0, t1, ..., tf
    # we only need the integral up to tf, final timepoint
    # Uses a 1e-6 pre-multiplier for the reg_strength to avoid wildcard issues in snakemake
    return 1e-6 * reg_strength * (l2_norm_nn_contrib[-1]/timespan_length)**2


def compute_discontinuity_penalty(
        solutions: Array,
        # cl_stage: int,
        state_dimension: int,
        reg_strength: float,
        masks: Array,
):
    """
    Compute the discontinuity penalty by measuring the sum of square errors
    between the last state of each shooting window and the first state of the next.
    """
    # TODO - If comparing with Development_Docker, reinstate MSE instead of SSE!
    if solutions.shape[0] > 1:  # valid for MS or for MS+CL before last stage
        sse = get_masked_sse(
            solutions[:-1, :, :state_dimension],  # endpoint
            solutions[1:, :, :state_dimension],  # startpoint
            masks[:-1].astype(int) & masks[1:].astype(int)
        )
    else:  # for MS+CL final CL stage (single valid window)
        sse = 0
    return reg_strength * sse


def _init_glorot(
        scale: float,
        shape: tuple,
        key: Any,  # TODO @Giacomo some weird error with KeyArray import with these package versions?!
):
    # TODO @Giacomo: add option for mode / distribution
    init = jnn.initializers.variance_scaling(scale, "fan_avg", "uniform")
    return init(key, shape)


def _is_linear(x):
    return isinstance(x, eqx.nn.Linear)


def _get_bias(m):
    return [x.bias for x in jtu.tree_leaves(m, is_leaf=_is_linear) if _is_linear(x)]


def _get_weight(m):
    return [x.weight for x in jtu.tree_leaves(m, is_leaf=_is_linear) if _is_linear(x)]


# TODO @GiacomoFabrini - add unit test
def set_weight_bias(
        mlp: Union[eqx.nn.MLP, eqx.Module],
        scale: float,
        key: Any,
) -> Union[eqx.nn.MLP, eqx.Module]:
    """
    Set the bias to zero, and use a glorot_uniform with provided scale for the
    weights
    """

    weight_old = _get_weight(mlp)
    bias_old = _get_bias(mlp)
    weight_new = [_init_glorot(scale, w.shape, key) for w in weight_old]
    mlp = eqx.tree_at(_get_weight, mlp, weight_new)
    # bias_new = [jnp.zeros(b.shape) for b in bias_old]
    # Change bias initialisation to small random values to avoid NaNs in hessian computation
    if mlp.use_bias:
        bias_new = [jr.normal(subkey, shape=b.shape) * scale for b, subkey in zip(bias_old, jr.split(key, len(bias_old)))]
        mlp = eqx.tree_at(_get_bias, mlp, bias_new)
    return mlp

def bounded_tanh_transform_params(
        params: Union[Array, jnp.ndarray],
        lb: tuple,
        ub: tuple,
        loc: float = 1.73,   # same as in Philipps et al. 2024
) -> jnp.ndarray:
    """
    Transforms `params` from linear scale to bounded tanh/arctanh scale (lower bounds: `lb`, upper bounds: `ub`).
    `loc` shifts the center of the tanh function.
    """
    # For numerical stability, shift the bounds slightly // https://www.biorxiv.org/content/10.1101/2024.11.29.626122v1
    lb = np.array([max(0.0, item * (1 - 1e-6)) for item in lb])
    ub = np.array(ub) * (1 + 1e-6)
    return np.arctanh(2 * (params - lb) / (ub - lb) - 1) + loc

def bounded_tanh_recover_params(
        params: Union[Array, jnp.ndarray],
        lb: tuple,
        ub: tuple,
        loc: float = 1.73,  # same as in Philipps et al. 2024
) -> jnp.ndarray:
    """
    Inverse-transforms `params` from bounded tanh/arctanh scale to linear one (lower bounds: `lb`, upper bounds: `ub`).
    `loc` shifts the center of the tanh function.
    """
    # For numerical stability, shift the bounds slightly // https://www.biorxiv.org/content/10.1101/2024.11.29.626122v1
    lb = jnp.array(lb) * (1 - 1e-6)
    ub = jnp.array(ub) * (1 + 1e-6)
    return lb + (np.tanh(params - loc) + 1) * (ub - lb) / 2


def process_mechanistic_params(
        dynamic_parameters: Union[Array, jnp.ndarray],
        options: dict[str, Union[bool, dict[str, Any]]]
) -> jnp.ndarray:
    """
    Process `dynamic_parameters` based on the scale specified in the `options` dictionary, returning
    the (possibly) transformed parameters `mech_params`.
    """
    if options["dynamic_params_scale"] == "log10":
        mech_params = jnp.power(10, dynamic_parameters)
    elif options["dynamic_params_scale"] == "linear":
        mech_params = dynamic_parameters
    elif (options["dynamic_params_scale"] == "bounded_tanh") and ("bounded_tanh_params" in options.keys()):
        mech_params = bounded_tanh_recover_params(
            params=dynamic_parameters,
            **options["bounded_tanh_params"],
        )
    else:
        raise ValueError(f"Unknown dynamic parameter scale: {options['dynamic_params_scale']}!")
    return mech_params
