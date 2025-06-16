import equinox as eqx
import fides
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import wandb

from diffrax import SaveAt
from functools import partial
from jaxtyping import Array, PyTree
from src.common import Conf, dynamic_param_labels
from src.model_training.training_util import transform_dynamic_params, prep_bfgs_training
from src.ude_problem import UDEProblem
from types import SimpleNamespace
from typing import Callable, List, Optional, Union



def fides_loss_grad(
        flat_params: jnp.ndarray,
        unflatten_params: Callable,
        loss_grad_fn: Callable,
        prob_static: UDEProblem,
        t_save_windows: jnp.ndarray,
        masks: jnp.ndarray,
        data,
        last_valid_index: int,
        mlp_reg_exp: float,
        mlp_reg_strength: float,
        output_reg: float,
        multiple_shooting_penalty: float,
        shooting_mode: str,
        max_steps: int
) -> tuple[float, np.ndarray]:
    # Unflatten params to compute loss and grads
    prob_params = unflatten_params(flat_params)
    (loss, _), grads= loss_grad_fn(
        prob_params, prob_static, t_save_windows, masks, data, last_valid_index,
        mlp_reg_exp, mlp_reg_strength, output_reg,
        multiple_shooting_penalty, shooting_mode, max_steps=max_steps
    )
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
        u0_val: object,
        t_save_windows: jnp.ndarray, masks: jnp.ndarray, training_data: jnp.ndarray,
        t_save_val: Union[SaveAt, Array], t_span_val: List, validation_data: jnp.ndarray,
        last_valid_index: int,
        loss_fn: Callable,
        val_loss_fn: Callable,
        zero_out_mlp: bool = False,
        wandb_log_final: bool = False,
        max_diffrax_steps: int = 20000
) -> tuple[UDEProblem, SimpleNamespace, UDEProblem, float, float, int, Optional[pd.DataFrame]]:

    model_fides = prep_bfgs_training(model, zero_out_mlp, bool(conf.output_reg or conf.mlp_reg_strength))

    # Split model_optx in learnable params and static component
    params, static = eqx.partition(model_fides, filter_spec)

    flat_params, unflatten_params = jax.flatten_util.ravel_pytree(params)

    fides_opt = fides.Optimizer(
        partial(
            fides_loss_grad,
            unflatten_params=unflatten_params,
            loss_grad_fn=loss_grad_fn,
            prob_static=static,
            t_save_windows=t_save_windows,
            masks=masks,
            data=training_data,
            last_valid_index=last_valid_index,
            mlp_reg_exp=conf.mlp_reg_exp,
            mlp_reg_strength=conf.mlp_reg_strength,
            output_reg=conf.output_reg,
            multiple_shooting_penalty=conf.multiple_shooting_penalty,
            shooting_mode=conf.shooting_mode,
            max_steps=max_diffrax_steps,
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
    except eqx.EquinoxRuntimeError:
        print("Could not optimise with fides - exceeded max_steps -> using initial params.")
        opt_params = params
    except RuntimeError as e:
        # catch RuntimeError in case of not finite initial loss, initial gradients (or initial gradient Hessian)
        # try to report on type of failure
        if "function" in e.__str__():
            print("Invalid function value - fides optimisation skipped.")
        elif "gradient entries" in e.__str__():
            print("Invalid gradients - fides optimisation skipped.")
        elif "gradient hessian" in e.__str__():
            print("Invalid hessian estimate - fides optimisation skipped.")
        # In any case, keep the initial params as opt_params -- fides could not optimise
        opt_params = params

    # Compute training loss
    loss_train, _ = loss_fn(
        prob_params=opt_params,
        prob_static=static,
        t_save_windows=t_save_windows,
        masks=masks,
        data=training_data,
        last_valid_index=last_valid_index,
        mlp_reg_exp=conf.mlp_reg_exp,
        mlp_reg_strength=conf.mlp_reg_strength,
        output_reg=conf.output_reg,
        multiple_shooting_penalty=conf.multiple_shooting_penalty,
        shooting_mode=conf.shooting_mode,
        max_steps=max_diffrax_steps
    )

    # Compute validation loss
    loss_val, _ = val_loss_fn(
        prob_params=opt_params,
        prob_static=static,
        u0=u0_val,
        t_save=t_save_val,
        t_span=t_span_val,
        data=validation_data,
        max_steps=2*max_diffrax_steps  # simulates for 2x the number of timepoints
    )

    # Get final model
    model = eqx.combine(opt_params, static)

    # create optimistix-like sol object, where value contains the optimised prob_params (dynamic part of the model)
    sol = SimpleNamespace(value=opt_params, stats={"num_steps": fides_opt.iteration})

    fides_results_df = None
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
        fides_results_df = pd.DataFrame(
            [{
                "optimiser": "fides_BFGS",
                "epoch": epoch + sol.stats["num_steps"],
                # Losses
                "loss_train": loss_train,
                "loss_val": loss_val,
                # Dynamic parameters
                **{
                    param: transform_dynamic_params(
                        model.func.dynamic_parameters, conf
                    )[i]
                    for i, param in enumerate(dynamic_param_labels[conf.dynamical_model])
                },
            }]
        )

    # Update epoch
    epoch += sol.stats["num_steps"]

    return model, sol, static, loss_train, loss_val, epoch, fides_results_df  # last can be None
