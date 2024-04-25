import equinox as eqx
import git
import itertools as itt
import jax.numpy as jnp
import jax
import numpy as np
import optax
import wandb

from common import Conf, EarlyStoppingParams
from flax.training.early_stopping import EarlyStopping
from jaxtyping import Array, Float, PyTree
from setup_diffrax_model import UDEProblem, compute_loss_grad, compute_loss, compute_corr
from typing import Dict, List


# Optimisers to choose from
optimisers = {
    "adam": optax.adam,
    "adamw": optax.adamw,
}


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

    # Initialise wandb and define metrics
    initialise_wandb(
        conf=conf,
        early_stopping_params=early_stopping_params,
        schedule_config=schedule_config,
        model=model,
    )

    # Optimiser and related schedule defined here
    # TODO @GiacomoFabrini: enable tunable schedule options
    schedule = optax.linear_schedule(**schedule_config)
    opt = optimisers[conf["optimiser"]](schedule)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

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

    @eqx.filter_jit
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
        return model, opt_state, loss_value, grads

    # Training loop
    for epoch in range(n_epochs + 1):
        model, opt_state, loss_train, grads = make_step(
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

        # Log loss values on training and validation dataset + learning rate (useful to inspect schedules)
        wandb.log(
            {
                "loss_train": loss_train,
                "loss_val": loss_val,
                "learning_rate": schedule(epoch),
            },
            step=epoch
        )

        # log param values and grads every 5 epochs
        if epoch % 5 == 0:
            wandb.log(
                {
                    **log_layer_stats(model.func.mlp, grads.func.mlp),
                },
                step=epoch
            )

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


# from Fabian's NeuralCoarseGraining
def log_layer_stats(
        model: eqx.nn.MLP,
        grad: eqx.nn.MLP,
):
    """
    Log the layer statistics to wandb.

    :param model: model containing weight information
    :param grad: model containing gradient information
    """
    layers = [
        (par_layer, grad_layer)
        for par_layer, grad_layer in zip(
            model.layers, grad.layers, strict=False
        )
        if isinstance(par_layer, eqx.nn.Linear)
    ]
    stats = {
        f'layer{ilayer}.{par_label}_{value_label}': wandb.Histogram(
            np.log10(np.abs(np.array(par_vals[par_vals != 0])))
        )
        if value_label == 'grads'
        else wandb.Histogram(list(par_vals))
        for ilayer, (par_layer, grad_layer) in enumerate(layers)
        for value_label, layer in zip(
            ('vals', 'grads'), (par_layer, grad_layer), strict=False
        )
        for par_label, par_vals in zip(
            ('w', 'b'),
            (layer.weight.ravel(), layer.bias.ravel()),
            strict=False,
        )
    }
    return stats


def initialise_wandb(
        conf: Conf,
        schedule_config: Dict,
        early_stopping_params: EarlyStoppingParams,
        model: UDEProblem,
):
    # Link to repo
    repo = git.Repo(search_parent_directories=True)
    # Initialise run
    wandb.init(
        project=f"TestUDE.{conf['dynamical_model']}",
        group=f"{conf['dynamical_model']}",
        config={
            **conf,
            "use_early_stopping": conf["use_early_stop"],  # early-stopping enabled/disabled
            "patience": early_stopping_params.patience
            if conf["use_early_stop"] else None,
            "min_improvement": early_stopping_params.min_improvement
            if conf["use_early_stop"] else None,
            "schedule_config": schedule_config,
            "optimizer": conf["optimiser"],
            "scheduler": "linear",
        },
        name="__".join(
            str(tag)
            for tag in (
                conf["dynamical_model"],
                conf["mlp_layer_width"],
                conf["mlp_depth"],
                conf["init_scale"],
                conf["activation_fn_name"],
                conf["optimiser"],
                conf["optimiser_learning_rate"],
            )
        ),
        settings=wandb.Settings(
            start_method="fork",
            git_commit=repo.head.object.hexsha,
            git_remote_url=repo.remotes.origin.url,
        ),
        tags=[
            "early_stop" if conf["use_early_stop"] else "no_early_stop",
        ]
    )

    # Define some metrics
    metrics = {
        "loss_train": "min",
        "loss_val": "min",
        "learning_rate": None,
        "patience_counter": None,
    }
    for metric in metrics.keys():
        # if metric summary not specified
        if metrics[metric] is None:
            wandb.define_metric(metric)
        else:
            wandb.define_metric(metric, summary=metrics[metric])
    # Define layer statistics (weight/bias value and grads)
    # from Fabian's NeuralCoarseGraining
    for il, _ in enumerate(model.func.mlp.layers):
        for par_label, val_label in itt.product(('w', 'b'), ('vals', 'grads')):
            wandb.define_metric(
                f'layer{il}.{par_label}_{val_label}',
                summary='last',
            )
