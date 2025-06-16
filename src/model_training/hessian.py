import jax
import jax.numpy as jnp

from src.common import Conf
from functools import partial
from jaxtyping import PyTree
from src.model_training.training_util import get_hessian_metrics, make_grads_finite
from src.model_training.losses import compute_hess_array
from src.ude_problem import UDEProblem
from typing import Callable, Union


# HESSIAN HELPER FUNCTIONS
def get_hessian(
        params: Union[PyTree, UDEProblem],
        static: Union[PyTree, UDEProblem],
        args: tuple,
        conf: Conf,
        base_loss_fn: Callable,
):
    # define hessian functions for different dynamical models
    hessian_fn = partial(compute_hess_array, loss_fn=base_loss_fn)
    # unpack args
    t_save, t_span, input_data, last_valid_index = args
    # flatten params PyTree and get unflatten function
    flat_p, unflatten_p = jax.flatten_util.ravel_pytree(params)
    # compute hessian as a 2D array
    hessian = hessian_fn(
        # equinox.debug.backward_nan(flat_p), unflatten_p, static,
        flat_p, unflatten_p, static,
        t_save, t_span, input_data, last_valid_index,
        conf.mlp_reg_exp, conf.mlp_reg_strength, conf.output_reg, conf.multiple_shooting_penalty,
        conf.shooting_mode
    )
    return hessian
