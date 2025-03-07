import jax
import jax.numpy as jnp

from src.common import Conf
from functools import partial
from jaxtyping import PyTree
from src.model_training.training_util import get_hessian_metrics, make_grads_finite
from src.model_training.losses import compute_hess_array, compute_batch_hess_array
from src.ude_problem import UDEProblem
from typing import Union


# HESSIAN HELPER FUNCTIONS
def get_hessian(
        params: Union[PyTree, UDEProblem],
        static: Union[PyTree, UDEProblem],
        args: tuple,
        conf: Conf,
):
    # define hessian functions for different dynamical models
    hessian_fn = compute_hess_array if conf.shooting_mode != "window_batch" else compute_batch_hess_array
    # unpack args
    u0, t_save, t_span, input_data = args
    # flatten params PyTree and get unflatten function
    flat_p, unflatten_p = jax.flatten_util.ravel_pytree(params)
    # compute hessian as a 2D array
    hessian = hessian_fn(
        # equinox.debug.backward_nan(flat_p), unflatten_p, static,
        flat_p, unflatten_p, static,
        u0, t_save, t_span, input_data,
        conf.mlp_reg_exp, conf.mlp_reg_strength, conf.output_reg, conf.multiple_shooting_penalty,
        conf.shooting_mode
    )
    return hessian


def apply_hessian_preconditioning_to_grads(
        grads: Union[PyTree, UDEProblem],
        params: Union[PyTree, UDEProblem],
        static: Union[PyTree, UDEProblem],
        args: tuple,
        conf: Conf,
        method="cg",  # alternative: "cg" for conjugate gradient, "leastsquares" for least squares
):
    # Compute the Hessian and metrics for current epoch
    hessian = get_hessian(params, static, args, conf)
    hessian_metrics = get_hessian_metrics(hessian, return_eigs_svs=False)

    # Flatten the grads PyTree
    flat_grads, unravel_grads = jax.flatten_util.ravel_pytree(grads)

    # Compute the Hessian-vector product function (Hv)
    def hessian_vec_product(hessian, vec):
        return jnp.dot(hessian, vec)

    if method in ["cg", "gmres"]:
        method_func = jax.scipy.sparse.linalg.cg if method == "cg" else jax.scipy.sparse.linalg.gmres
        # Use GMRES or Conjugate Gradient to solve H @ x = grads
        flat_precond_grads, _ = method_func(
            partial(hessian_vec_product, hessian), flat_grads
        )
    elif method == "leastsquares":
        # Compute the inverse of the Hessian
        # inv_hessian = jnp.linalg.pinv(hessian)
        # Solve the linear system H @ X = I, where I is the identity matrix of the same size as the hessian, H
        flat_precond_grads = jax.scipy.linalg.solve(hessian, flat_grads)
    else:
        raise ValueError(f"Unknown preconditioning method: {method}!")
    # Unflatten the preconditioned gradients back to the PyTree structure + ensure they are finite
    grads = make_grads_finite(unravel_grads(flat_precond_grads))
    return grads, hessian_metrics
