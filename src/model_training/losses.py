import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

from src.model_training.model_util import compute_discontinuity_penalty, regularise_mlp, regularise_mlp_contribution
from src.model_training.training_util import (get_sol_and_loss_single_shooting, get_sol_and_loss_multiple_shooting,
                                              get_sol_and_loss_validation)
from typing import Callable



# @eqx.filter_jit
def compute_loss(
        prob_params, prob_static,
        u0, t_save, t_span,
        data,
        mlp_reg_exp: float,
        mlp_reg_strength: float,
        output_reg: float,
        multiple_shooting_penalty: float,
        shooting_mode: str,
        loss_fn: Callable,
        eval_mode: bool = False,
        adjoint: diffrax.AbstractAdjoint = diffrax.RecursiveCheckpointAdjoint()  # default for everything apart from Hessian (DirectAdjoint)
):
    # Get full UDEProblem
    prob = eqx.combine(prob_params, prob_static)

    if shooting_mode == "single":
        sol, loss = get_sol_and_loss_single_shooting(prob, adjoint, loss_fn, data)
        discontinuity_penalty = 0
    elif shooting_mode == "multiple":
        sol, loss = get_sol_and_loss_multiple_shooting(prob, adjoint, data)
        # Average losses and add discontinuity penalty
        discontinuity_penalty = compute_discontinuity_penalty(
            sol=sol,
            cl_stage=prob.curriculum_stage,
            state_dimension=prob_static.func.state_dimension,
            reg_strength=multiple_shooting_penalty,
            masks=jax.lax.stop_gradient(prob.masks)
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
        if shooting_mode == "single":  # "window_batch" calls vmapped single-shooting
            mlp_output_reg = regularise_mlp_contribution(
                l2_norm_nn_contrib=sol.ys[:, -1],
                timespan_length=(t_save.subs.ts[-1] - t_save.subs.ts[0]) if eval_mode  # validation
                    else jax.lax.stop_gradient(prob.t_save_windows[0][-1]-prob.t_save_windows[0][0]),  # training
                reg_strength=output_reg
            )
        elif shooting_mode == "multiple":
            valid_windows = jax.lax.stop_gradient(prob.t_save_windows) if prob.curriculum_stage < 2 \
                else jax.lax.stop_gradient(prob.t_save_windows)[:-(prob.curriculum_stage - 1), :]
            # TODO @GiacomoFabrini - should this be a sum or a mean?
            mlp_output_reg = jnp.sum(jnp.array(
                [
                    regularise_mlp_contribution(
                        l2_norm_nn_contrib=sol.ys[i, :, -1],
                        timespan_length=(t_save_window[-1] - t_save_window[0]),
                        reg_strength=output_reg
                    )
                    for i, t_save_window in enumerate(jax.lax.stop_gradient(valid_windows))
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
        u0, t_save, t_span,
        data,
        mlp_reg_exp, mlp_reg_strength, output_reg, multiple_shooting_penalty,
        shooting_mode,
        loss_fn: Callable,
        adjoint: diffrax.AbstractAdjoint = diffrax.RecursiveCheckpointAdjoint()
):
    return eqx.filter_value_and_grad(compute_loss, has_aux=True)(
        prob_params, prob_static,
        u0, t_save, t_span,
        data,
        mlp_reg_exp, mlp_reg_strength, output_reg, multiple_shooting_penalty, shooting_mode, loss_fn,
        eval_mode=False, adjoint=adjoint
    )


def compute_validation_loss(
        prob_params, prob_static,
        u0, t_save, t_span,
        data,
        loss_fn: Callable,
        adjoint: diffrax.AbstractAdjoint = diffrax.RecursiveCheckpointAdjoint()  # default for everything apart from Hessian (DirectAdjoint)
):
    # Get full UDEProblem
    prob = eqx.combine(prob_params, prob_static)
    sol, loss = get_sol_and_loss_validation(prob, u0, t_save, t_span, adjoint, loss_fn, data)
    # only interested in MSE, not in mlp_reg or output_reg -> return a None reg_loss_dict
    return loss, None


# Optimistix loss needs to return a None aux
def compute_loss_optx(prob_params, args):
    (prob_static, u0, t_save, t_span, data,
     mlp_reg_exp, mlp_reg_strength, output_reg,
     multiple_shooting_penalty, shooting_mode, eval_mode, loss_fn) = args
    loss, _ = compute_loss(
        prob_params, prob_static,
        u0, t_save, t_span,
        data,
        mlp_reg_exp, mlp_reg_strength, output_reg, multiple_shooting_penalty, shooting_mode,
        loss_fn,
        eval_mode,
    )
    return loss, None


# Optimistix loss needs to return a None aux
def compute_val_loss_optx(prob_params, args):
    (prob_static, u0, t_save, t_span, data, loss_fn) = args
    loss, _ = compute_validation_loss(
        prob_params, prob_static,
        u0, t_save, t_span,
        data,
        loss_fn,
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
        u0, t_save, t_span, data,
        mlp_reg_exp, mlp_reg_strength, output_reg,
        multiple_shooting_penalty, shooting_mode,
        loss_fn,
        adjoint=diffrax.DirectAdjoint()
):
    prob_params = unflatten_p(flat_p)
    loss, _ = compute_loss(
        prob_params, prob_static,
        u0, t_save, t_span,
        data,
        mlp_reg_exp, mlp_reg_strength, output_reg, multiple_shooting_penalty,
        shooting_mode,
        loss_fn,
        eval_mode=False,
        adjoint=adjoint,
    )
    return loss


@eqx.filter_jit
def compute_batch_loss(
        prob_params, prob_static,
        u0, t_save, t_span, data,
        mlp_reg_exp, mlp_reg_strength, output_reg,
        loss_fn, **kwargs  # TODO kwargs needed or can be removed?
):
    losses, mlp_output_regs = jax.vmap(compute_loss, in_axes=(None, None, 0, 0, 0, 0, None, None, None, None, None, None))(
        prob_params, prob_static,
        u0, t_save, t_span, data,
        mlp_reg_exp, mlp_reg_strength, output_reg, 0, "single", loss_fn,
        False
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
        mlp_reg_exp, mlp_reg_strength, output_reg, multiple_shooting_penalty, shooting_mode, loss_fn
):
    return eqx.filter_value_and_grad(compute_batch_loss, has_aux=True)(
        prob_params, prob_static,
        u0, t_save, t_span, data,
        mlp_reg_exp, mlp_reg_strength, output_reg,
        multiple_shooting_penalty=multiple_shooting_penalty,
        shooting_mode=shooting_mode,
        loss_fn=loss_fn,
        eval_mode=False,
    )

# BATCH LOSS OPTIMISTIX/FIDES
def compute_batch_loss_optx(prob_params, args):
    (prob_static, u0, t_save, t_span, data,
     mlp_reg_exp, mlp_reg_strength, output_reg,
     multiple_shooting_penalty, shooting_mode, loss_fn) = args
    loss, _ = compute_batch_loss(
        prob_params, prob_static,
        u0, t_save, t_span,
        data,
        mlp_reg_exp, mlp_reg_strength, output_reg,
        multiple_shooting_penalty=multiple_shooting_penalty,
        shooting_mode=shooting_mode,
        loss_fn=loss_fn,
        eval_mode=False,
    )
    return loss, None


@eqx.filter_jit
@jax.hessian
def compute_batch_hess_array(
        flat_p, unflatten_p,
        prob_static,
        u0, t_save, t_span, data,
        mlp_reg_exp, mlp_reg_strength, output_reg,
        multiple_shooting_penalty, shooting_mode,
        loss_fn: Callable,
        adjoint=diffrax.DirectAdjoint()
):
    prob_params = unflatten_p(flat_p)
    loss, _ = compute_batch_loss(
        prob_params, prob_static,
        u0, t_save, t_span,
        data,
        mlp_reg_exp, mlp_reg_strength, output_reg,
        multiple_shooting_penalty=multiple_shooting_penalty,
        shooting_mode=shooting_mode,
        loss_fn=loss_fn,
        eval_mode=False,
        adjoint=adjoint,
    )
    return loss
