import dataclasses
import equinox as eqx
import jax.numpy as jnp
import jax.nn as jnn
import jax


def _init_glorot(scale, shape, key):
    init = jnn.initializers.variance_scaling(scale, "fan_avg", "uniform")
    return init(key, shape)


def set_weight_bias(mlp, scale, key):
    """
    Set the bias to zero, and use a glorot_uniform with provided scale for the 
    weights
    """
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    _get_bias = lambda m: [x.bias
                           for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                           if is_linear(x)]
    _get_weight = lambda m: [x.weight
                             for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                             if is_linear(x)]

    weight_old = _get_weight(mlp)
    bias_old = _get_bias(mlp)
    weight_new = [_init_glorot(scale, w.shape, key) for w in weight_old]
    bias_new = [jnp.zeros(b.shape) for b in bias_old]

    mlp = eqx.tree_at(_get_bias, mlp, bias_new)
    mlp = eqx.tree_at(_get_weight, mlp, weight_new)
    return mlp


@dataclasses.dataclass
class Conf(dict):
    dynamical_model: str
    mlp_layer_width: int
    mlp_depth: int
    activation_fn_name: str = "relu"
    optimiser_learning_rate: float = 1e-3



@dataclasses.dataclass
class EarlyStoppingParams(dict):
    use_early_stopping: bool = True
    patience: int = 9
    min_improvement: float = 0