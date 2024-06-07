import equinox as eqx
import git
import itertools as itt
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import wandb

from pathlib import Path
from src.common import Conf, EarlyStoppingParams
from src.ude_problem import UDEProblem
from typing import List


# Scheduler from Fabian's NeuralCoarseGraining
def get_scheduler(
        conf: Conf,
        n_epochs: int,
):
    """Get the learning rate scheduler.

    Parameters
    ----------
    conf : configuration object
    n_epochs : int - total number of training epochs

    Returns
    ----------
    optax.sgdr_schedule
        The learning rate scheduler.
    """
    epochs_per_schedule = np.array([
        conf.opt_steps * (conf.opt_mult ** i)
        for i in range(int(n_epochs // conf.opt_steps))
        if conf.opt_steps * (conf.opt_mult ** i) <= n_epochs
    ])
    schedules = [
        {
            'init_value': conf.max_lrate / conf.lrate_span * conf.lrate_decay ** i_schedule,
            'peak_value': conf.max_lrate * conf.lrate_decay ** i_schedule,
            'warmup_steps': int(
                (conf.opt_steps * (conf.opt_mult ** i_schedule))
                * conf.warmup_fct
            ),
            'decay_steps': int(conf.opt_steps * (conf.opt_mult ** i_schedule)),
            'end_value': conf.max_lrate / conf.lrate_span * conf.lrate_decay ** (i_schedule + 1),
        }
        for i_schedule in range(len(epochs_per_schedule))
    ]
    return optax.sgdr_schedule(schedules)


def initialise_wandb(
        conf: Conf,
        early_stopping_params: EarlyStoppingParams,
        model: UDEProblem,
):
    # Link to repo
    repo = git.Repo(search_parent_directories=True)
    # Initialise run
    wandb.init(
        project=f"TestUDE.{conf.dynamical_model}",
        group=f"{conf.dynamical_model}",
        config={
            **conf.__dict__,
            "patience": early_stopping_params.patience
            if conf.use_early_stop else None,
            "min_improvement": early_stopping_params.min_improvement
            if conf.use_early_stop else None,
        },
        name='__'.join(field for field in conf.__str__().split('__')[1:]),  # dynamical_system info already in project
        settings=wandb.Settings(
            start_method="fork",
            git_commit=repo.head.object.hexsha,
            git_remote_url=repo.remotes.origin.url,
        ),
        tags=[
            "early_stop" if conf.use_early_stop else "no_early_stop",
            "frozen_mechanistic_params" if conf.freeze_mechanistic_params else "learnable_mechanistic_params",
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


# from Fabian's NeuralCoarseGraining
def log_model_stats(
        model: UDEProblem,
        grad: UDEProblem,
        conf: Conf,
):
    """
    Log the layer statistics to wandb.

    :param model: model containing weight information
    :param grad: model containing gradient information
    """
    layers = [
        (par_layer, grad_layer)
        for par_layer, grad_layer in zip(
            model.func.mlp.layers, grad.func.mlp.layers, strict=False
        )
        if isinstance(par_layer, eqx.nn.Linear)
    ]
    layer_stats = {
        f'layer{ind_layer}.{par_label}_{value_label}': wandb.Histogram(
            np.log10(np.abs(np.array(par_vals[par_vals != 0])))
        )
        if value_label == 'grads'
        else wandb.Histogram(list(par_vals))
        for ind_layer, (par_layer, grad_layer) in enumerate(layers)
        for value_label, layer in zip(
            ('vals', 'grads'), (par_layer, grad_layer), strict=False
        )
        for par_label, par_vals in zip(
            ('w', 'b'),
            (layer.weight.ravel(), layer.bias.ravel()),
            strict=False,
        )
    }

    if conf.dynamical_model == 'lotka_volterra':
        param_labels = ['alpha', 'delta']
    elif conf.dynamical_model == 'sir':
        param_labels = ['beta', 'gamma', 'delta']
    else:
        raise ValueError(f"Unknown dynamical model: {conf.dynamical_model}!")

    if not conf.freeze_mechanistic_params:
        dynamic_param_stats = {
            f'dynamic_param.{par_label}_{value_label}': wandb.Histogram(
                np.log10(np.abs(np.array(par_val[par_val != 0])))
            )
            if value_label == 'grads'
            else wandb.Histogram(par_val)
            for value_label, par_vals in zip(
                ('vals', 'grads'),
                (model.func.dynamic_parameters, grad.func.dynamic_parameters),
                strict=False
            )
            for par_label, par_val in zip(
                param_labels,
                np.array(par_vals).ravel(),
                strict=False
            )
        }
    else:
        # No gradients to log! (frozen dynamical parameters)
        dynamic_param_stats = {
            f'dynamic_param.{par_label}_{value_label}':  wandb.Histogram(par_val)
            for value_label, par_vals in zip(
                'vals',
                model.func.dynamic_parameters,
                strict=False
            )
            for par_label, par_val in zip(
                param_labels,
                np.array(par_vals).ravel(),
                strict=False
            )
        }

    stats = {**layer_stats, **dynamic_param_stats}
    return stats


def log_plots(
        df_predict: pd.DataFrame,
        df_obs: pd.DataFrame,
        labels: List[str],
        colors: List[str],
        epoch: int,
        plot_name: str,
        plot_dir: Path,
        save_plot: bool = False,
        plot_fmt: str = "png"
):
    plt.subplots(1, 2, figsize=(6, 4), sharey=True, sharex=True)
    for ind, (label, color) in enumerate(zip(labels, colors)):
        plt.subplot(1, 2, ind + 1)
        plt.plot(
            df_predict['time'].values,
            df_predict[label].values,
            color=color,
            ls='--',
            label=f'model_{label}',
        )
        plt.plot(
            df_obs['time'].values,
            df_obs[label].values,
            color=color,
            ls='',
            marker='o',
            label=f'obs_{label}',
        )
        plt.legend(frameon=False)
        plt.xlabel('Time')
        plt.gca().tick_params(direction='in', length=5)
        if ind == 0:
            plt.ylabel('Measurement')
    if save_plot and (plot_dir is not None):
        plt.savefig(plot_dir / f"fit_plot_{plot_name}.{plot_fmt}")
    elif save_plot and (plot_dir is None):
        raise ValueError("Must specify plot directory in order to save plots!")
    else:
        wandb.log({f"fit_plot_{plot_name}": plt}, step=epoch)
