import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
import pandas as pd
import wandb

from src.common import Conf, dynamic_param_labels
from diffrax import SaveAt
from jaxtyping import PyTree
from src.model_training.training_util import log_model_stats, transform_dynamic_params, prep_bfgs_training
from src.ude_problem import UDEProblem
from types import SimpleNamespace
from typing import Callable, List, Optional


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
        u0_train: object,
        u0_val: object,
        t_save_train: SaveAt, t_span_train: List, training_data: jnp.ndarray,
        t_save_val: SaveAt, t_span_val: List, validation_data: jnp.ndarray,
        frozen_mechanistic: bool = False, frozen_nn: bool = False,
        zero_out_mlp: bool = False,
        wandb_log: bool = True,
        wandb_log_final: bool = True,
) -> tuple[UDEProblem, SimpleNamespace, UDEProblem, float, float, int, Optional[pd.DataFrame]]:

    # Define options and tags for BFGS solver
    options = {
        "rtol": optx_config["rtol"],
        "atol": optx_config["atol"],
    }
    tags = frozenset()
    # Additional training with BFGS via optimistix
    solver = optx.BFGS(**options)

    model_optx = prep_bfgs_training(model, zero_out_mlp)

    # Split model_optx in learnable params and static component
    params, static = eqx.partition(model_optx, filter_spec)

    # Define additional arguments to pass optimistix loss function (train and val) for step by step solve
    args = (
        static, u0_train, t_save_train, t_span_train, training_data,
        conf.mlp_reg_exp, conf.mlp_reg_strength, conf.output_reg,
        conf.multiple_shooting_penalty, conf.shooting_mode, False,  # not in eval
        base_loss
    )
    args_val = (
        static, u0_val, t_save_val, t_span_val, validation_data,
        val_base_loss
    )
    # Shape+dtype of the output of `fn`
    f_struct = jax.ShapeDtypeStruct((), jnp.float64)
    aux_struct = None

    # Optimistix solve
    sol, results_df, loss_train, loss_val = optx_solve(
        y=params, solver=solver,
        fn_train=loss_optx, fn_val=val_loss_optx, args=args, args_val=args_val,
        f_struct=f_struct, aux_struct=aux_struct,
        options=options, tags=tags,
        max_steps=num_epochs,
        starting_epoch=epoch,
        conf=conf,
        frozen_mechanistic=frozen_mechanistic,
        frozen_nn=frozen_nn,
        wandb_log=wandb_log,
        wandb_log_final=wandb_log_final,
    )

    # Get final model
    model = eqx.combine(sol.value, static)
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

    return model, sol, static, loss_train, loss_val, epoch, results_df  # last can be None


def optx_solve(
        y, solver, fn_train, fn_val,
        args, args_val,
        f_struct, aux_struct,
        options, tags, max_steps,
        starting_epoch, conf,
        frozen_mechanistic, frozen_nn,
        wandb_log: bool = True,
        wandb_log_final: bool = True,
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
    while not (done or iteration > max_steps):
        # print(f"Evaluating point {y} with value {fn(y, args)}.")
        y, state, aux = step(y=y, state=state)
        # Log training and validation loss + params and grads every 5 optimistix iterations
        if wandb_log and (iteration % 5 == 0):
            loss_train = fn_train(y, args)[0]
            loss_val = fn_val(y, args_val)[0]
            wandb.log(
                {
                    "loss_train": loss_train,
                    "loss_val": loss_val,
                },
                step=starting_epoch+iteration
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
                            param: transform_dynamic_params(
                                eqx.combine(y, args[0])  # params + static = model
                                .func.dynamic_parameters, conf
                            )[i]
                            for i, param in enumerate(dynamic_param_labels[conf.dynamical_model])
                        },
                    }]
                )
            )
            # Only log model stats if training loss is finite (can compute derivatives)
            if jnp.isfinite(loss_train):
                wandb.log(
                    log_model_stats(
                        eqx.combine(y, args[0]),  # recombine updated params and static
                        state.f_info.grad,
                        conf,
                        frozen_mechanistic, frozen_nn,
                    ),
                    step=starting_epoch+iteration
                )
        iteration += 1
        done, result = terminate(y=y, state=state)
    if result != optx.RESULTS.successful:
        print(f"Oh no! Got error {result}.")
    y, _, _ = solver.postprocess(fn_train, y, aux, args, options, state, tags, result)
    # print(f"Found solution {y} with value {fn(y, args)}.")

    # Log final training and validation loss terms + dynamic parameter values for final BFGS fine-tuning (third stage)
    if wandb_log_final:
        wandb.log(
            {
                **{
                    f"final_{param}_BFGS": transform_dynamic_params(
                        eqx.combine(y, args[0]).func.dynamic_parameters, conf
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
    if wandb_log:
        return SimpleNamespace(**sol), pd.concat(results_dfs), fn_train(y, args)[0], fn_val(y, args_val)[0]
    else:
        return SimpleNamespace(**sol), None, fn_train(y, args)[0], fn_val(y, args_val)[0]
