import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

from functools import partial
from jaxtyping import PyTree
from src.common import Conf
from src.model_training.regularisation import (
    compute_discontinuity_penalty,
    regularise_mlp,
    regularise_mlp_contribution
)
from src.model_training.training_util import (
    get_sol_and_loss_single_shooting,
    get_sol_and_loss_multiple_shooting,
    get_sol_and_loss_validation
)
from src.ude_problem import UDEProblem
from typing import Callable, Union


@eqx.filter_jit
def compute_loss(
        prob_params, prob_static,
        t_save_windows, masks,
        data,
        last_valid_index: int,
        mlp_reg_exp: float,
        mlp_reg_strength: float,
        output_reg: float,
        multiple_shooting_penalty: float,
        shooting_mode: str,
        loss_fn: Callable,
        adjoint: diffrax.AbstractAdjoint = diffrax.RecursiveCheckpointAdjoint(), # default for everything apart from Hessian (DirectAdjoint)
        max_steps: int = 20000,
):
    # Get full UDEProblem
    prob = eqx.combine(prob_params, prob_static)

    if shooting_mode == "single":
        sol, loss = get_sol_and_loss_single_shooting(
            prob,
            adjoint,
            loss_fn,
            data,
            t_save_windows,
            masks,
            max_steps
        )
        discontinuity_penalty = 0
    elif shooting_mode == "multiple":
        sol, loss = get_sol_and_loss_multiple_shooting(
            prob,
            adjoint,
            loss_fn,
            data,
            t_save_windows,
            masks,
            last_valid_index,
            max_steps
        )
        # Average losses and add discontinuity penalty
        discontinuity_penalty = compute_discontinuity_penalty(
            solutions=sol.ys[:last_valid_index],
            state_dimension=prob_static.func.state_dimension,
            reg_strength=multiple_shooting_penalty,
            masks=masks[:last_valid_index],
        )
        loss +=  discontinuity_penalty
    else:
        raise ValueError(f"Unknown shooting mode: {shooting_mode}!")

    # Add MLP and output regularisation
    if prob.func.regularise_network:
        mlp_reg = regularise_mlp(
            mlp=prob.func.mlp,
            reg_exp=mlp_reg_exp,
            reg_strength=mlp_reg_strength
        )
        if shooting_mode == "single":
            mlp_output_reg = regularise_mlp_contribution(
                l2_norm_nn_contrib=sol.ys[:, -1],
                timespan_length=t_save_windows[0][-1]-t_save_windows[0][0],
                reg_strength=output_reg
            )
        elif shooting_mode == "multiple":
            mlp_output_reg = jnp.mean(jnp.array(
                [
                    regularise_mlp_contribution(
                        l2_norm_nn_contrib=sol.ys[i, :, -1],
                        timespan_length=t_save_windows[i][-1] - t_save_windows[i][0],
                        reg_strength=output_reg
                    )
                    for i in range(last_valid_index)
                ]
            ))
        else:
            raise ValueError("This function should only be called with 'single' or 'multiple' shooting_mode!")
        loss += (mlp_reg + mlp_output_reg)
        return loss, {
            "discontinuity_penalty": discontinuity_penalty,
            "mlp_reg": mlp_reg,
            "nn_contrib_reg": mlp_output_reg
        }
    return loss, None


def compute_loss_grad(
        prob_params, prob_static,
        t_save_windows, masks,
        data,
        last_valid_index,
        mlp_reg_exp, mlp_reg_strength, output_reg, multiple_shooting_penalty,
        shooting_mode,
        loss_fn: Callable,
        adjoint: diffrax.AbstractAdjoint = diffrax.RecursiveCheckpointAdjoint(),
        max_steps: int = 20000,
):
    return eqx.filter_value_and_grad(compute_loss, has_aux=True)(
        prob_params, prob_static,
        t_save_windows, masks,
        data,
        last_valid_index,
        mlp_reg_exp, mlp_reg_strength, output_reg, multiple_shooting_penalty, shooting_mode, loss_fn,
        adjoint=adjoint, max_steps=max_steps
    )


def compute_validation_loss(
        prob_params, prob_static,
        u0, t_save, t_span,
        data,
        loss_fn: Callable,
        adjoint: diffrax.AbstractAdjoint = diffrax.RecursiveCheckpointAdjoint(),  # default for everything apart from Hessian (DirectAdjoint),
        max_steps: int = 40000  # 2x as long for 2x the timecourse
):
    # Get full UDEProblem
    prob = eqx.combine(prob_params, prob_static)
    sol, loss = get_sol_and_loss_validation(prob, u0, t_save, t_span, adjoint, loss_fn, data, max_steps)
    # only interested in MSE, not in mlp_reg or output_reg -> return a None reg_loss_dict
    return loss, None


# Optimistix loss needs to return a None aux
def compute_loss_optx(prob_params, args):
    (prob_static, t_save_windows, masks, data, last_valid_index,
     mlp_reg_exp, mlp_reg_strength, output_reg,
     multiple_shooting_penalty, shooting_mode, loss_fn, max_steps) = args
    loss, _ = compute_loss(
        prob_params, prob_static,
        t_save_windows, masks,
        data,
        last_valid_index,
        mlp_reg_exp, mlp_reg_strength, output_reg, multiple_shooting_penalty, shooting_mode,
        loss_fn, max_steps=max_steps
    )
    return loss, None


# Optimistix loss needs to return a None aux
def compute_val_loss_optx(prob_params, args):
    (prob_static, u0, t_save, t_span, data, loss_fn, max_steps) = args
    loss, _ = compute_validation_loss(
        prob_params, prob_static,
        u0, t_save, t_span,
        data,
        loss_fn, max_steps=max_steps
    )
    return loss, None


@eqx.filter_jit
@jax.hessian
def compute_hess_array(
        flat_p, unflatten_p,
        prob_static,
        t_save_windows, masks, data,
        last_valid_index,
        mlp_reg_exp, mlp_reg_strength, output_reg,
        multiple_shooting_penalty, shooting_mode,
        loss_fn,
        adjoint=diffrax.DirectAdjoint(),
        max_steps: int = 20000
):
    prob_params = unflatten_p(flat_p)
    loss, _ = compute_loss(
        prob_params, prob_static,
        t_save_windows, masks,
        data,
        last_valid_index,
        mlp_reg_exp, mlp_reg_strength, output_reg, multiple_shooting_penalty,
        shooting_mode,
        loss_fn,
        adjoint=adjoint,
        max_steps=max_steps
    )
    return loss


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
