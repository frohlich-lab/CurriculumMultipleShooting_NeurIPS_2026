import time
from functools import partial
from types import SimpleNamespace
from typing import Callable, Optional

import equinox as eqx
import fides
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import wandb
from jaxtyping import PyTree

from src.common import Conf, dynamic_param_labels
from src.model_training.model_util import process_mechanistic_params
from src.model_training.training_util import prep_bfgs_training
from src.ude_problem import UDEProblem


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
        t_save_windows: jnp.ndarray, masks: jnp.ndarray, training_data: jnp.ndarray,
        last_valid_index: int,
        loss_fn: Callable,
        rescaling_args: dict,
        zero_out_mlp: bool = False,
        wandb_log: bool = False,
        max_diffrax_steps: int = 20000
) -> tuple[UDEProblem, SimpleNamespace, UDEProblem, int, Optional[pd.DataFrame]]:

    # Initialise variables
    opt_wall_duration, opt_process_duration, iterations = jnp.nan, jnp.nan, 0

    # Prepare model and split into learnable/static components
    model_fides = prep_bfgs_training(model, zero_out_mlp, bool(conf.output_reg or conf.mlp_reg_strength))
    params, static = eqx.partition(model_fides, filter_spec)

    flat_params, unflatten_params = jax.flatten_util.ravel_pytree(params)

    # Compute gradient at starting parameters to get starting gradient sup-norm
    _, flat_grads = fides_loss_grad(
        flat_params,
        unflatten_params,
        loss_grad_fn,
        static,
        t_save_windows,
        masks,
        training_data,
        last_valid_index,
        conf.mlp_reg_exp,
        conf.mlp_reg_strength,
        conf.output_reg,
        conf.multiple_shooting_penalty,
        conf.shooting_mode,
        max_diffrax_steps
    )
    inv_grad_supnorm = 1 / jnp.linalg.norm(flat_grads, ord=jnp.inf)

    # Initialise BFGS Hessian with rescaled identity matrix as per Rackauckas et al.
    # ref: https://github.com/ChrisRackauckas/universal_differential_equations/blob/master/LotkaVolterra/scenario_1.jl#L118
    initial_hessian = 0.01 * inv_grad_supnorm * np.eye(flat_params.shape[0])

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
        options={"maxiter": max_iter, "maxtime": 2 * 3600}
    )

    try:
        # Optimise (and time)
        wall_t0, process_t0 = time.perf_counter(), time.process_time()
        opt_f, opt_x, opt_grad, opt_hess = fides_opt.minimize(
            np.array(flat_params),  # move flat_params to numpy array to avoid JAX immutability
            hess0=initial_hessian,
        )
        wall_t1, process_t1 = time.perf_counter(), time.process_time()
        opt_wall_duration, opt_process_duration = wall_t1 - wall_t0, process_t1 - process_t0
        iterations = fides_opt.iteration

        # Unflatten optimised parameters into UDEProblem-like opt_params
        opt_params = unflatten_params(opt_x)
    # TODO - does this do anything or can we safely remove it?
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

    # Get final model
    model = eqx.combine(opt_params, static)

    # create optimistix-like sol object, where value contains the optimised prob_params (dynamic part of the model)
    sol = SimpleNamespace(value=opt_params, stats={"num_steps": fides_opt.iteration})

    fides_results_df = pd.DataFrame(
        [{
            "optimiser": "fides_BFGS",
            "epoch": epoch + sol.stats["num_steps"],
            # Losses
            "loss_train": loss_train,
            # Dynamic parameters
            **{
                f"raw_{param}": process_mechanistic_params(
                    model.func.dynamic_parameters,
                    model.func.options
                )[i]
                for i, param in enumerate(dynamic_param_labels[conf.dynamical_model])
            },
            **{
                param: process_mechanistic_params(
                    model.func.dynamic_parameters,
                    model.func.options
                )[i] / rescaling_args["time_rescaling"]  # ensure actual parameters are properly rescaled
                for i, param in enumerate(dynamic_param_labels[conf.dynamical_model])
            },
            "bfgs_iterations": iterations,
            "bfgs_wall_duration": opt_wall_duration,
            "bfgs_process_duration": opt_process_duration,
        }]
    )

    # Log final training and validation loss terms + dynamic parameter values (rescaled)
    if wandb_log:
        wandb.log(
            {
                **{
                    f"final_{param}_BFGS": process_mechanistic_params(
                        model.func.dynamic_parameters,
                        model.func.options
                    )[i] / rescaling_args["time_rescaling"]
                    for i, param in enumerate(dynamic_param_labels[conf.dynamical_model])
                },
                "final_loss_train_BFGS": loss_train,
            },
            step=epoch + sol.stats["num_steps"]
        )

    # Update epoch
    epoch += sol.stats["num_steps"]

    return model, sol, static, epoch, fides_results_df  # last can be None
