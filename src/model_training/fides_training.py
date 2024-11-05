import equinox as eqx
import fides
import jax
import jax.numpy as jnp
import numpy as np
import wandb

from diffrax import SaveAt
from functools import partial
from jaxtyping import PyTree
from src.common import Conf, dynamic_param_labels
from src.model_training.model_util import set_weight_bias
from src.model_training.training_util import transform_dynamic_params
from src.ude_problem import UDEProblem
from types import SimpleNamespace
from typing import Callable, List



def fides_loss_grad(
        flat_params: jnp.ndarray,
        unflatten_params: Callable,
        loss_grad_fn: Callable,
        prob_static: UDEProblem,
        u0: jnp.ndarray,
        t_save: SaveAt,
        t_span,
        data,
        mlp_reg_exp: float,
        mlp_reg_strength: float,
        output_reg: float
) -> tuple[float, np.ndarray]:
    # Unflatten params to compute loss and grads
    prob_params = unflatten_params(flat_params)
    (loss, _), grads= loss_grad_fn(prob_params, prob_static, u0, t_save, t_span, data, mlp_reg_exp, mlp_reg_strength, output_reg)
    # Flatten grads and cast to numpy array to avoid JAX immutability
    flat_grads = np.array(jax.flatten_util.ravel_pytree(grads)[0])
    # Return loss, grad -- Hessian will be computed with approximation (BFGS)
    return float(loss), flat_grads


def train_fides(
        model: UDEProblem,
        filter_spec: PyTree,
        conf: Conf,
        loss_grad_fn: Callable,
        epoch: int,
        max_iter: int,
        u0: object,
        t_save_train: SaveAt, t_span_train: List, training_data: jnp.ndarray,
        t_save_val: SaveAt, t_span_val: List, validation_data: jnp.ndarray,
        loss_fn: Callable,
        zero_out_mlp: bool = False,
        wandb_log_final: bool = False
) -> tuple[UDEProblem, SimpleNamespace, UDEProblem, float, float, int]:

    # Zero out MLP if needed (mechanistic fit only)
    if zero_out_mlp:
        mlp_replacement = set_weight_bias(model.func.mlp, 0.0, jax.random.PRNGKey(0))
    else:
        mlp_replacement = model.func.mlp
    # Set dropout behaviour to be deterministic
    model_fides = model.copy(
        new_mlp=eqx.nn.inference_mode(mlp_replacement, value=True)
    )

    # Split model_optx in learnable params and static component
    params, static = eqx.partition(model_fides, filter_spec)

    flat_params, unflatten_params = jax.flatten_util.ravel_pytree(params)

    fides_opt = fides.Optimizer(
        partial(
            fides_loss_grad,
            unflatten_params=unflatten_params,
            loss_grad_fn=loss_grad_fn,
            prob_static=static,
            u0=u0,
            t_save=t_save_train,
            t_span=t_span_train,
            data=training_data,
            mlp_reg_exp=conf.mlp_reg_exp,
            mlp_reg_strength=conf.mlp_reg_strength,
            output_reg=conf.output_reg,
        ),
        ub=np.asarray([jnp.inf for _ in flat_params]),
        lb=np.asarray([-jnp.inf for _ in flat_params]),
        hessian_update=fides.hessian_approximation.BFGS(),
        options={"maxiter": max_iter},
    )

    try:
        # Optimise
        opt_f, opt_x, opt_grad, opt_hess = fides_opt.minimize(
            np.array(flat_params)  # move flat_params to numpy array to avoid JAX immutability
        )
        # Unflatten optimised parameters into UDEProblem-like opt_params
        opt_params = unflatten_params(opt_x)
    except RuntimeError as e:
        # catch RuntimeError in case of not finite initial loss
        loss_train, _ = loss_fn(
            prob_params=params,
            prob_static=static,
            u0=u0,
            t_save=t_save_train,
            t_span=t_span_train,
            data=training_data,
            mlp_reg_exp=conf.mlp_reg_exp,
            mlp_reg_strength=conf.mlp_reg_strength,
            output_reg=conf.output_reg,
        )
        # In that case, keep the initial params as opt_params -- fides could not optimise
        if not np.isfinite(loss_train):
            print("Warning: fides could not optimise due to initial non-finite loss - reporting unoptimised parameters!")
            opt_params = params
        else:
            raise ValueError(f"finite loss, but encountered exception {e} in fides minimisation")

    # Compute training loss
    loss_train, _ = loss_fn(
        prob_params=opt_params,
        prob_static=static,
        u0=u0,
        t_save=t_save_train,
        t_span=t_span_train,
        data=training_data,
        mlp_reg_exp=conf.mlp_reg_exp,
        mlp_reg_strength=conf.mlp_reg_strength,
        output_reg=conf.output_reg,
    )

    # Compute validation loss
    loss_val, _ = loss_fn(
        prob_params=opt_params,
        prob_static=static,
        u0=u0,
        t_save=t_save_val,
        t_span=t_span_val,
        data=validation_data,
        mlp_reg_exp=conf.mlp_reg_exp,
        mlp_reg_strength=conf.mlp_reg_strength,
        output_reg=conf.output_reg,
    )

    # Get final model
    model = eqx.combine(opt_params, static)

    # create optimistix-like sol object
    sol = SimpleNamespace(value=opt_params, stats={"num_steps": fides_opt.iteration})

    # Log final training and validation loss terms + dynamic parameter values for final BFGS fine-tuning (third stage)
    if wandb_log_final:
        wandb.log(
            {
                **{
                    f"final_{param}_BFGS": transform_dynamic_params(
                        model.func.dynamic_parameters, conf
                    )[i]
                    for i, param in enumerate(dynamic_param_labels[conf.dynamical_model])
                },
                "final_loss_train_BFGS": loss_train,
                "final_loss_val_BFGS": loss_val,
            },
            step=epoch + sol.stats["num_steps"]
        )


    # Update epoch
    epoch += sol.stats["num_steps"]

    # Plot eigenvalue and singular value spectrum
    # hessian = get_hessian(sol.value, static, (u0, t_save_train, t_span_train, training_data), conf)
    # hessian_metrics, eigs, svs = get_hessian_metrics(hessian, return_eigs_svs=True)
    # eig_sv_plot(
    #     eigs, svs,
    #     conf,
    #     epoch,
    # )
    return model, sol, static, loss_train, loss_val, epoch
