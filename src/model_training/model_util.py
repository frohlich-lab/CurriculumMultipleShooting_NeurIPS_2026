import equinox as eqx
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Array
from typing import Any, Union


def get_mse(pred: Array, data: Array):
    return jnp.mean((pred - data) ** 2)


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
    # the l2_norm_nn_contrib is the integral of the 2-norm of the nn_contribution from t0 to t0, t1, ..., tf
    # we only need the integral up to tf, final timepoint
    return reg_strength * (l2_norm_nn_contrib[-1]/timespan_length)**2


def _init_glorot(
        scale: float,
        shape: tuple,
        key: Any,  # TODO @Giacomo some weird error with KeyArray import with these package versions?!
):
    init = jnn.initializers.variance_scaling(scale, "fan_avg", "uniform")
    return init(key, shape)


def _is_linear(x):
    return isinstance(x, eqx.nn.Linear)


def _get_bias(m):
    return [x.bias for x in jtu.tree_leaves(m, is_leaf=_is_linear) if _is_linear(x)]


def _get_weight(m):
    return [x.weight for x in jtu.tree_leaves(m, is_leaf=_is_linear) if _is_linear(x)]


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
    # bias_new = [jnp.zeros(b.shape) for b in bias_old]
    # Change bias initialisation to small random values to avoid NaNs in hessian computation
    bias_new = [jr.normal(subkey, shape=b.shape) * scale for b, subkey in zip(bias_old, jr.split(key, len(bias_old)))]

    mlp = eqx.tree_at(_get_bias, mlp, bias_new)
    mlp = eqx.tree_at(_get_weight, mlp, weight_new)
    return mlp
