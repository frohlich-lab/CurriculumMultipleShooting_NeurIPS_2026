import diffrax
import equinox as eqx
import git
import jax
import jax.numpy as jnp
import jax.random as jr
import jax
import numpy as np
import pandas as pd
import os
import wandb

import lotka_volterra.setup_diffrax_model as setup_lv

from common import Conf, EarlyStoppingParams
from flax.training.early_stopping import EarlyStopping
from jaxtyping import Array, Float, PyTree
from optax import adam, linear_schedule
from setup_diffrax_model import UDEProblem, compute_loss_grad, compute_loss
from typing import Dict, List


def train(
        model: UDEProblem,
        u0: jnp.ndarray,
        training_data: jnp.ndarray,
        t_save_train: List,
        t_span_train: List,
        validation_data: jnp.ndarray,
        t_save_val: List,
        t_span_val: List,
        conf: Conf,
        schedule_config: Dict,
        n_epochs: int,
        early_stopping_params: EarlyStoppingParams,
) -> UDEProblem:
    """
    Trains the UDEmodel (`model`).

    Parameters
    ----------
    :param model:
        UDEProblem to train.
    :param u0:
        starting conditions (prey, predator) for the model.
    :param t_save_train:
        time points to save the solution at for training.
    :param t_save_val:
        time points to save the solution at for validation.
    :param t_span_train:
        time span to solve the UDE for training.
    :param t_span_val:
        time span to solve the UDE for validation.
    :param training_data:
        training data for the model.
    :param validation_data:
        validation data for the model.
    :param conf:
        configuration (hyperparameters).
    :param schedule_config:
        optimiser schedule.
    :param n_epochs:
        Number of training iterations (epochs).
    :param early_stopping_params:
        Dictionary of parameters for Early-Stopping.

    """

    repo = git.Repo(search_parent_directories=True)

    wandb.init(
        project=f"TestUDE.{conf['dynamical_model']}",
        group=f"{conf['dynamical_model']}",
        config={
            **conf,
            "use_early_stopping": early_stopping_params.use_early_stopping,  # early-stopping enabled/disabled
            "patience": early_stopping_params.patience
            if early_stopping_params.use_early_stopping else None,
            "min_improvement": early_stopping_params.min_improvement
            if early_stopping_params.use_early_stopping else None,
            "schedule_config": schedule_config,
            "optimizer": "adam",
            "scheduler": "linear",
        },
        name="__".join(
            str(tag)
            for tag in (
                conf["dynamical_model"],
                conf["mlp_layer_width"],
                conf["mlp_depth"],
                conf["activation_fn_name"],
                conf["optimiser_learning_rate"],
            )
        ),
        settings=wandb.Settings(
            start_method="fork",
            git_commit=repo.head.object.hexsha,
            git_remote_url=repo.remotes.origin.url,
        ),
        tags=[
            "early_stop" if early_stopping_params.use_early_stopping else "no_early_stop",
        ]
    )

    # Define W&B metrics in modular fashion
    metrics = {
        "loss_train": "min",
        "loss_val": "min",
        "patience_counter": None,
    }
    for metric in metrics.keys():
        # if metric summary not specified
        if metrics[metric] is None:
            wandb.define_metric(metric)
        else:
            wandb.define_metric(metric, summary=metrics[metric])

    # Optimiser and related schedule defined here
    # TODO @GiacomoFabrini: shall we tune optimiser and schedule? (later)
    schedule = linear_schedule(**schedule_config)
    opt = adam(schedule)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    # loss_val_min = np.inf

    # Check Early-stopping parameters have been set correctly and instantiate early stopper
    if early_stopping_params.use_early_stopping:
        if early_stopping_params.patience is None:
            raise ValueError("Patience value for early stopping is undefined.")
        elif early_stopping_params.min_improvement is None:
            raise ValueError("Minimum absolute improvement for early stopping is undefined.")
        else:
            early_stopper = EarlyStopping(
                min_delta=early_stopping_params.min_improvement,
                patience=early_stopping_params.patience
            )

    # TODO @GiacomoFabrini restore jitting once debugged
    #@eqx.filter_jit
    def make_step(
            model: UDEProblem,
            opt_state: PyTree,
            input_data: Float[Array, '...'],  # TODO @GiacomoFabrini fix input data shape
            t_save: List,
            t_span: List,
    ):
        loss_value, grads = compute_loss_grad(
            model,
            u0,
            t_save,
            t_span,
            input_data
        )
        grads = jax.tree_map(
            lambda x: jnp.where(jnp.isfinite(x), x, jnp.zeros_like(x)),
            grads,
        )
        updates, opt_state = opt.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    # Training loop
    for epoch in range(n_epochs + 1):
        model, opt_state, loss_train = make_step(
            model=model,
            opt_state=opt_state,
            input_data=training_data,
            t_save=t_save_train,
            t_span=t_span_train,
        )

        # compute validation loss
        loss_val = compute_loss(
            model,
            u0,
            t_save_val,
            t_span_val,
            validation_data,
        )

        # Log loss values on training and validation dataset
        wandb.log(
            {
                "loss_train": loss_train,
                "loss_val": loss_val,
            },
            step=epoch
        )
        #
        # if loss_val < loss_val_min:
        #     loss_val_min = loss_val

        if early_stopping_params.use_early_stopping:
            # Update early stopper
            early_stopper = early_stopper.update(loss_val)
            # Debugging statements
            print(
                f"epoch {epoch} | "
                f"loss_val {loss_val} | "
                f"has improved? {early_stopper.has_improved} | "
                f"patience count {early_stopper.patience_count}"
            )
            # Log current patience count
            wandb.log(
                {
                    "patience_counter": early_stopper.patience_count,
                },
                step=epoch,
            )
            # Stop training if we have run out of patience
            if early_stopper.should_stop:
                print(f'Met early stopping criteria, breaking at epoch {epoch}')
                break

    wandb.log({"final_epoch": epoch})
    wandb.finish()
    return model
