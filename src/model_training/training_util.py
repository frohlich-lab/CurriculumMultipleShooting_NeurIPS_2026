import equinox as eqx
import jax.numpy as jnp
import jax.nn as jnn
import jax.tree_util as jtu

from typing import Any, Union


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


def _init_glorot(
        scale: float,
        shape: tuple,
        key: Any,  # TODO @Giacomo some weird error with KeyArray import with these package versions?!
):
    init = jnn.initializers.variance_scaling(scale, "fan_avg", "uniform")
    return init(key, shape)


def set_weight_bias(
        mlp: Union[eqx.nn.MLP, eqx.Module],
        scale: float,
        key: Any,
) -> Union[eqx.nn.MLP, eqx.Module]:
    """
    Set the bias to zero, and use a glorot_uniform with provided scale for the
    weights
    """
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    _get_bias = lambda m: [x.bias
                           for x in jtu.tree_leaves(m, is_leaf=is_linear)
                           if is_linear(x)]
    _get_weight = lambda m: [x.weight
                             for x in jtu.tree_leaves(m, is_leaf=is_linear)
                             if is_linear(x)]

    weight_old = _get_weight(mlp)
    bias_old = _get_bias(mlp)
    weight_new = [_init_glorot(scale, w.shape, key) for w in weight_old]
    bias_new = [jnp.zeros(b.shape) for b in bias_old]

    mlp = eqx.tree_at(_get_bias, mlp, bias_new)
    mlp = eqx.tree_at(_get_weight, mlp, weight_new)
    return mlp
