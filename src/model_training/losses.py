import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from functools import partial
from src.model_examples.lotka_volterra.setup_diffrax_model import lv_base_loss as compute_loss_lv
from src.model_examples.sir.setup_diffrax_model import sir_base_loss as compute_loss_sir
from src.model_training.model_util import (compute_discontinuity_penalty, get_mse,
                                           regularise_mlp, regularise_mlp_contribution)


# TODO @GiacomoFabrini no magic numbers - add conf.node_augmentation and derive number of components for loss.
#  Same for all functions on this page! Need to adapt SIR as well!
@eqx.filter_jit
def compute_loss(
        prob_params, prob_static,
        u0, t_save, t_span,
        data,
        mlp_reg_exp, mlp_reg_strength, output_reg, multiple_shooting_penalty,
        shooting_mode: str,
        adjoint=None
):
    # Get full UDEProblem
    prob = eqx.combine(prob_params, prob_static)

    # Get model-specific loss function
    if prob.dynamical_model == "lotka_volterra":
        loss_fn = partial(compute_loss_lv, state_dimension=prob.func.state_dimension)
    elif prob.dynamical_model == "sir":
        loss_fn = compute_loss_sir
    else:
        raise ValueError(f"Unknown dynamical model: {prob.dynamical_model}!")

    # Handle single-/multiple-shooting
    if shooting_mode == "single":
        sol = prob.solve_single_shooting(diffrax.Tsit5(), u0, t_save, t_span, adjoint=adjoint)
        loss = loss_fn(sol=sol, data=data)
    elif shooting_mode == "multiple":
        sol = prob.solve(diffrax.Tsit5(), u0, t_save, t_span, adjoint=adjoint)
        if not isinstance(t_save, diffrax.SaveAt):
            t_save = diffrax.SaveAt(ts=t_save)
        num_windows = int(np.ceil(len(t_save.subs.ts) / (prob.shooting_window_length - 1)))
        losses = jnp.array([
            get_mse(
                pred=solution.ys[:, :prob.func.state_dimension],
                data=data[i*(prob.shooting_window_length-1) : (i+1)*prob.shooting_window_length-i]
            )
            for solution, i in zip(sol, range(num_windows))
        ])
        # Average losses and add discontinuity penalty
        loss = (jnp.mean(losses)
                + multiple_shooting_penalty * compute_discontinuity_penalty(prob, sol, num_windows))
    else:
        raise ValueError(f"Unknown shooting mode: {shooting_mode}!")

    # Add MLP and output regularisation
    if prob.func.regularise_network:
        if mlp_reg_strength > 0:
            mlp_reg = regularise_mlp(
                mlp=prob.func.mlp,
                reg_exp=mlp_reg_exp,
                reg_strength=mlp_reg_strength
            )
        else:
            mlp_reg = 0
        if output_reg > 0:
            if shooting_mode == "single":
                mlp_output_reg = regularise_mlp_contribution(
                    l2_norm_nn_contrib=sol.ys[:, prob.func.state_dimension],
                    timespan_length=t_span[-1]-t_span[0],
                    reg_strength=output_reg
                )
            else:
                # TODO @GiacomoFabrini NEED TO FIX THIS! -- discuss with Fabian during Jour Fixe
                mlp_output_reg = regularise_mlp_contribution(
                    l2_norm_nn_contrib=sol[0].ys[:, prob.func.state_dimension],  # this is only using the first segment!
                    timespan_length=t_span[-1]-t_span[0],
                    reg_strength=output_reg
                )
        else:
            mlp_output_reg = 0
        loss += (mlp_reg + mlp_output_reg)
        return loss, mlp_output_reg
    return loss, None


def compute_loss_grad(
        prob_params, prob_static,
        u0, t_save, t_span,
        data,
        mlp_reg_exp, mlp_reg_strength, output_reg, multiple_shooting_penalty,
        shooting_mode
):
    return eqx.filter_value_and_grad(compute_loss, has_aux=True)(
        prob_params, prob_static,
        u0, t_save, t_span,
        data,
        mlp_reg_exp, mlp_reg_strength, output_reg, multiple_shooting_penalty, shooting_mode,
        adjoint=None
    )


# Optimistix loss needs to return a None aux
def compute_loss_optx(prob_params, args):
    (prob_static, u0, t_save, t_span, data,
     mlp_reg_exp, mlp_reg_strength, output_reg, multiple_shooting_penalty, shooting_mode) = args
    loss, _ = compute_loss(
        prob_params, prob_static,
        u0, t_save, t_span,
        data,
        mlp_reg_exp, mlp_reg_strength, output_reg, multiple_shooting_penalty,shooting_mode,
        None
    )
    return loss, None


# HESSIAN -- UNUSED
# @eqx.filter_jit
# @jax.hessian
# def compute_hess(prob_params, prob_static, u0, t_save, t_span, data, adjoint):
#     prob = eqx.combine(prob_params, prob_static)
#     u_pred = prob.solve(diffrax.Tsit5(), u0, t_save, t_span,
#                         adjoint=adjoint)
#     return jnp.mean((u_pred.ys[:, :prob.func.state_dimension] - data) ** 2)


@eqx.filter_jit
@jax.hessian
def compute_hess_array(
        flat_p, unflatten_p,
        prob_static,
        u0, t_save, t_span, data, adjoint,
        mlp_reg_exp, mlp_reg_strength, output_reg, multiple_shooting_penalty,
        shooting_mode
):
    prob_params = unflatten_p(flat_p)
    loss, _ = compute_loss(
        prob_params, prob_static,
        u0, t_save, t_span,
        data,
        mlp_reg_exp, mlp_reg_strength, output_reg, multiple_shooting_penalty,
        shooting_mode,
        adjoint=adjoint,
    )
    return loss


@eqx.filter_jit
def compute_batch_loss(
        prob_params, prob_static,
        u0, t_save, t_span, data,
        mlp_reg_exp, mlp_reg_strength, output_reg, adjoint=None, **kwargs
):
    losses, mlp_output_regs = jax.vmap(compute_loss, in_axes=(None, None, 0, 0, 0, 0, None, None, None, None, None, None))(
        prob_params, prob_static,
        u0, t_save, t_span, data,
        mlp_reg_exp, mlp_reg_strength, output_reg, 0, "single", adjoint
    )
    mean_loss = jnp.mean(losses)
    if eqx.combine(prob_params, prob_static).func.regularise_network:
        mean_mlp_output_reg = jnp.mean(mlp_output_regs)
        return mean_loss, mean_mlp_output_reg
    return mean_loss, None


# LOSS AND GRAD
def compute_batch_loss_grad(
        prob_params, prob_static,
        u0, t_save, t_span, data,
        mlp_reg_exp, mlp_reg_strength, output_reg, multiple_shooting_penalty, shooting_mode
):
    return eqx.filter_value_and_grad(compute_batch_loss, has_aux=True)(
        prob_params, prob_static,
        u0, t_save, t_span, data,
        mlp_reg_exp, mlp_reg_strength, output_reg,
        multiple_shooting_penalty=multiple_shooting_penalty,
        shooting_mode=shooting_mode,
        adjoint=None
    )

# BATCH LOSS OPTIMISTIX/FIDES
def compute_batch_loss_optx(prob_params, args):
    (prob_static, u0, t_save, t_span, data,
     mlp_reg_exp, mlp_reg_strength, output_reg, multiple_shooting_penalty, shooting_mode) = args
    loss, _ = compute_batch_loss(
        prob_params, prob_static,
        u0, t_save, t_span,
        data,
        mlp_reg_exp, mlp_reg_strength, output_reg,
        multiple_shooting_penalty=multiple_shooting_penalty,
        shooting_mode=shooting_mode,
        adjoint=None
    )
    return loss, None


@eqx.filter_jit
@jax.hessian
def compute_batch_hess_array(
        flat_p, unflatten_p,
        prob_static,
        u0, t_save, t_span, data, adjoint,
        mlp_reg_exp, mlp_reg_strength, output_reg, multiple_shooting_penalty,
        shooting_mode
):
    prob_params = unflatten_p(flat_p)
    loss, _ = compute_batch_loss(
        prob_params, prob_static,
        u0, t_save, t_span,
        data,
        mlp_reg_exp, mlp_reg_strength, output_reg,
        multiple_shooting_penalty=multiple_shooting_penalty,
        shooting_mode=shooting_mode,
        adjoint=adjoint,
    )
    return loss
