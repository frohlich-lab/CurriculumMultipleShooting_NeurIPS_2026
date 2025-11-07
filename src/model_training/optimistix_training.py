import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
import pandas as pd
import time
import wandb

from src.common import Conf, dynamic_param_labels
from diffrax import SaveAt
from jaxlib.xla_extension import XlaRuntimeError
from jaxtyping import Array, PyTree

from src.model_training.model_util import process_mechanistic_params
from src.model_training.training_util import prep_bfgs_training
from src.ude_problem import UDEProblem
from types import SimpleNamespace
from typing import Callable, List, Optional, Union


# OPTIMISTIX OPTIMISATION FUNCTIONS
def train_optimistix(
        model: UDEProblem,
        filter_spec: PyTree,
        conf: Conf,
        optx_config: dict[str: float],
        base_loss: Callable,
        loss_optx: Callable,
        val_base_loss: Callable,
        val_loss_optx: Callable,
        epoch: int,
        num_epochs: int,
        u0_val: object,
        t_save_windows: Array,
        masks: Array,
        training_data: jnp.ndarray,
        t_save_val: Union[SaveAt, Array],
        t_span_val: List,
        validation_data: jnp.ndarray,
        last_valid_index: int,
        zero_out_mlp: bool = False,
        wandb_log: bool = True,
        max_diffrax_steps: int = 20000
) -> tuple[UDEProblem, SimpleNamespace, UDEProblem, float, float, int, Optional[pd.DataFrame]]:

    # Define options and tags for BFGS solver
    options = {
        "rtol": optx_config["rtol"],
        "atol": optx_config["atol"],
    }
    tags = frozenset()
    # Additional training with BFGS via optimistix
    solver = optx.BFGS(**options)

    model_optx = prep_bfgs_training(model, zero_out_mlp, bool(conf.output_reg or conf.mlp_reg_strength))

    # Split model_optx in learnable params and static component
    params, static = eqx.partition(model_optx, filter_spec)

    # Define additional arguments to pass optimistix loss function (train and val) for step by step solve
    args = (
        static, t_save_windows, masks, training_data, last_valid_index,
        conf.mlp_reg_exp, conf.mlp_reg_strength, conf.output_reg,
        conf.multiple_shooting_penalty, conf.shooting_mode,
        base_loss, max_diffrax_steps
    )
    args_val = (
        static, u0_val, t_save_val, t_span_val, validation_data,
        val_base_loss, 2*max_diffrax_steps  # simulates for 2x the number of timepoints
    )
    # Shape+dtype of the output of `fn`
    f_struct = jax.ShapeDtypeStruct((), jnp.float64)
    aux_struct = None

    # Optimistix solve
    try:
        sol, results_df, loss_train, loss_val = optx_solve(
            y=params, solver=solver,
            fn_train=loss_optx, fn_val=val_loss_optx, args=args, args_val=args_val,
            f_struct=f_struct, aux_struct=aux_struct,
            options=options, tags=tags,
            max_steps=num_epochs,
            starting_epoch=epoch,
            conf=conf,
            wandb_log=wandb_log,
        )
    except XlaRuntimeError as e:
        print(f"Optimistix ran into error: {e}")
        # If there is an error, we return the current model and stats
        sol = SimpleNamespace(value=params, stats={"num_steps": 0, "termination_reason": "error"})
        results_df = None
        loss_train, loss_val = jnp.inf, jnp.inf

    # Get final model
    model = eqx.combine(sol.value, static)
    # Update epoch
    epoch += sol.stats["num_steps"]

    return model, sol, static, loss_train, loss_val, epoch, results_df  # last can be None


def optx_solve(
        y, solver, fn_train, fn_val,
        args, args_val,
        f_struct, aux_struct,
        options, tags, max_steps,
        starting_epoch, conf,
        wandb_log: bool = True,
):
    # Initialise loss variables
    loss_train, loss_val = jnp.inf, jnp.inf
    aux = None

    # These arguments are always fixed throughout interactive solves.
    step = eqx.filter_jit(
        eqx.Partial(solver.step, fn=fn_train, args=args, options=options, tags=tags)
    )
    terminate = eqx.filter_jit(
        eqx.Partial(solver.terminate, fn=fn_train, args=args, options=options, tags=tags)
    )

    # Initial state before we start solving.
    state = solver.init(fn_train, y, args, options, f_struct, aux_struct, tags)
    done, result = terminate(y=y, state=state)

    iteration = 1
    results_dfs = []
    # Start timing
    wall_t0, process_t0 = time.perf_counter(), time.process_time()

    while not (done or iteration > max_steps):
        # print(f"Evaluating point {y} with value {fn(y, args)}.")
        y, state, aux = step(y=y, state=state)
        # Log training and validation loss + params and grads every 5 optimistix iterations
        if iteration % 10 == 0:
            loss_train = fn_train(y, args)[0]
            loss_val = fn_val(y, args_val)[0]
            # Include some logging (not automatic as in fides)
            print(
                f"epoch {starting_epoch+iteration} | "
                f"loss_train: {loss_train} | "
                f"loss_val: {loss_val}"
            )
            results_dfs.append(
                pd.DataFrame(
                    [{
                        "optimiser": "optimistix_BFGS",
                        "epoch": starting_epoch + iteration,
                        # Losses
                        "loss_train": loss_train,
                        "loss_val": loss_val,
                        # Dynamic parameters
                        **{
                            param: process_mechanistic_params(
                                eqx.combine(y, args[0]).func.dynamic_parameters,
                                eqx.combine(y, args[0]).func.options
                            )[i]
                            for i, param in enumerate(dynamic_param_labels[conf.dynamical_model])
                        },
                    }]
                )
            )
            if wandb_log:
                wandb.log(
                    {
                        "loss_train": loss_train,
                        "loss_val": loss_val,
                    },
                    step=starting_epoch + iteration
                )
        iteration += 1
        done, result = terminate(y=y, state=state)

    # Stop timing
    wall_t1, process_t1 = time.perf_counter(), time.process_time()
    opt_wall_duration, opt_process_duration = wall_t1 - wall_t0, process_t1 - process_t0

    if result != optx.RESULTS.successful:
        print(f"Oh no! Got error {result}.")
    y, _, _ = solver.postprocess(fn_train, y, aux, args, options, state, tags, result)
    # print(f"Found solution {y} with value {fn(y, args)}.")

    # Log final training and validation loss terms + dynamic parameter values for final BFGS fine-tuning (third stage)
    if wandb_log:
        wandb.log(
            {
                **{
                    f"final_{param}_BFGS": process_mechanistic_params(
                        eqx.combine(y, args[0]).func.dynamic_parameters,
                        eqx.combine(y, args[0]).func.options
                    )[i]
                    for i, param in enumerate(dynamic_param_labels[conf.dynamical_model])
                },
                "final_loss_train_BFGS": loss_train,
                "final_loss_val_BFGS": loss_val,
            },
            step=starting_epoch + iteration
        )

    sol = {
        "value": y,  # this is the optimised prob_params (dynamic part of the model)
        "stats": {
            "num_steps": iteration,
            "termination_reason": result,
        },
    }

    optx_results_df = pd.concat(results_dfs).assign(
        iterations=iteration,
        bfgs_wall_duration=opt_wall_duration,
        bfgs_process_duration=opt_process_duration
    )

    return SimpleNamespace(**sol), optx_results_df, fn_train(y, args)[0], fn_val(y, args_val)[0]
