import diffrax
import equinox as eqx
import git
import itertools as itt
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import seaborn as sns
import wandb

from jaxtyping import Array, PyTree
from pathlib import Path
from src.common import Conf, EarlyStoppingParams, plots_dir
from src.model_examples.lotka_volterra.setup_diffrax_model import compute_hess_array as compute_hess_array_lv
from src.model_examples.sir.setup_diffrax_model import compute_hess_array as compute_hess_array_sir
from src.model_training.training_configuration import N_EPOCHS, MAX_ITER_OPTX
from src.ude_problem import UDEProblem
from typing import List, Union


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


def get_opt_opt_state_params(
        conf: Conf,
        params: Union[PyTree, UDEProblem],
        schedule: optax.Schedule
):
    # Optimisers to choose from
    optimisers = {
        "adam": optax.adam(schedule),
        "adamw": optax.adamw(schedule),
        "adam_lbfgs": optax.chain(
            optax.adam(schedule),
            optax.scale_by_lbfgs()
        ),
        "sgd": optax.sgd(schedule),
        "sgd_lbfgs": optax.chain(
           optax.sgd(schedule),
           optax.scale_by_lbfgs()
        )
    }

    if not conf.use_lookahead:
        opt = optimisers[conf.optimiser]
        opt_state = opt.init(params)
    else:
        opt = optax.lookahead(
            fast_optimizer=optimisers[conf.optimiser],
            sync_period=conf.lookahead_sync,
            slow_step_size=conf.lookahead_step_size,
            reset_state=conf.lookahead_reset_state,
        )
        # Initialise params for lookahead (now params has params.fast and params.slow)
        params = optax.LookaheadParams.init_synced(params)
        opt_state = opt.init(params)

    return opt, opt_state, params


def initialise_wandb(
        conf: Conf,
        early_stopping_params: EarlyStoppingParams,
        model: UDEProblem,
):
    # Link to repo
    repo = git.Repo(search_parent_directories=True)
    # Initialise run
    wandb.init(
        project=f"TestUDE.nf.{conf.dynamical_model}",  # nf = nextflow pipeline
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
        "max_eigenvalue": None,
        "min_eigenvalue": None,
        "max_sv": None,
        "min_sv": None,
        "condition_number": "min",
        "max_grad_norm": "min",
        "num_train_samples": None,
        "patience_counter": None,
    }
    # Add regularisation term if applied
    if conf.reg_strength > 0:
        metrics["regularisation_loss"] = "min"

    # Define metrics in W&B
    for metric in metrics.keys():
        # if metric summary not specified
        if metrics[metric] is None:
            wandb.define_metric(metric)
        else:
            wandb.define_metric(metric, summary=metrics[metric])

    # Define layer statistics (weight/bias value and grads)
    # from Fabian's NeuralCoarseGraining
    # https://github.com/frohlich-lab/NeuralCoarseGraining/blob/8c5cc42447b9b55cdf8b3bfc685388ba4bf6b751/ncg/training.py#L257
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
    :param conf: configuration object
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


def make_grads_finite(grads):
    return jax.tree_map(
        lambda x: jnp.where(jnp.isfinite(x), x, jnp.zeros_like(x)),
        grads,
    )


def get_model_after_step(
        params: Union[PyTree, UDEProblem],
        static: Union[PyTree, UDEProblem],
        conf: Conf,
):
    if conf.use_lookahead:
        model = eqx.combine(params.fast, static)
    else:
        model = eqx.combine(params, static)
    return model


def plot_model_weights(model: UDEProblem, conf: Conf, filename: str = None):
    num_rows = conf.mlp_layer_width + 2  # top and bottom margin
    num_columns = len(model.func.mlp.layers) + 2  # left and right margin

    in_size = {
        'lotka_volterra': 2,
        'sir': 3,
    }

    out_size = {
        'lotka_volterra': 2,
        'sir': 1,
    }

    fig = plt.figure()
    fig.set_figheight(num_rows)
    fig.set_figwidth(num_columns)

    # Plot input
    ax = plt.subplot2grid(
        shape=(num_rows, num_columns),
        loc=(3, 0),
        colspan=1,
        rowspan=in_size[conf.dynamical_model],
    )
    sns.heatmap(
        np.zeros(shape=(in_size[conf.dynamical_model], 1)),
        cbar=False, xticklabels=False, yticklabels=False, ax=ax
    )
    ax.set_title(f"in: {in_size[conf.dynamical_model]}")
    for i, layer in enumerate(model.func.mlp.layers):
        ax = plt.subplot2grid(
            shape=(num_rows, num_columns),
            loc=(2, 1+i),
            colspan=1,
            rowspan=conf.mlp_layer_width,
        )
        sns.heatmap(layer.weight.T, cbar=False, xticklabels=False, yticklabels=False, ax=ax)
        ax.set_title(f"{i}: {layer.weight.T.shape}")

    # Plot output
    ax = plt.subplot2grid(
        shape=(num_rows, num_columns),
        loc=(3, 1+len(model.func.mlp.layers)),
        colspan=1,
        rowspan=out_size[conf.dynamical_model],
    )
    sns.heatmap(
        np.zeros(shape=(out_size[conf.dynamical_model], 1)),
        cbar=False, xticklabels=False, yticklabels=False, ax=ax
    )
    ax.set_title(f"out: {out_size[conf.dynamical_model]}")

    if filename is not None:
        plt.savefig(filename, facecolor='w')
    plt.show()


def generate_incremental_arrays(
        conf: Conf,
        data: Array,
        t_save: diffrax.SaveAt,
        n_epoch: int,
        increment_percentage: int,
):
    if increment_percentage <= 0 or increment_percentage > 100 or 100 % increment_percentage != 0:
        raise ValueError("Increment percentage must be a divisor of 100 (e.g. 10, 20, 25, 50).")

    # Initialize the list to store the results
    incremental_data = []
    incremental_t_save = []
    incremental_t_span = []

    if conf.curriculum_learning:
        for i in range(1, increment_percentage + 1):
            length = int(len(data) * (i / increment_percentage))
            start_epoch = int((i - 1) * (int(0.50 * n_epoch) / increment_percentage))
            end_epoch = int(i * (int(0.50 * n_epoch) / increment_percentage))
            for _ in range(end_epoch - start_epoch):
                incremental_data.append(data[:length])
                incremental_t_save.append(diffrax.SaveAt(ts=t_save.subs.ts[:length]))
                incremental_t_span.append([np.min(t_save.subs.ts[:length]), np.max(t_save.subs.ts[:length])])

        # Add full data for the last 50% of epochs
        for _ in range(n_epoch - len(incremental_data)):
            incremental_data.append(data)
            incremental_t_save.append(t_save)
            incremental_t_span.append([np.min(t_save.subs.ts), np.max(t_save.subs.ts)])
    else:
        incremental_data = [data] * n_epoch
        incremental_t_save = [t_save] * n_epoch
        incremental_t_span = [[np.min(t_save.subs.ts), np.max(t_save.subs.ts)]] * n_epoch

    return incremental_data, incremental_t_save, incremental_t_span


def eig_sv_plot(
        params: Union[PyTree, UDEProblem],
        static: Union[PyTree, UDEProblem],
        u0: Array,
        t_save: diffrax.SaveAt,
        t_span: List,
        data: jnp.ndarray,
        conf: Conf,
        epoch: int,
        plot: bool = True,
):
    # flatten params PyTree and get unflatten function
    flat_p_hess, unflatten_p_hess = jax.flatten_util.ravel_pytree(params)
    hessian_fn = {
        "lotka_volterra": compute_hess_array_lv,
        "sir": compute_hess_array_sir,
    }
    # compute hessian
    hessian = hessian_fn[conf.dynamical_model](
        flat_p_hess, unflatten_p_hess, static,
        u0, t_save, t_span, data,
        diffrax.DirectAdjoint(),
        conf.reg_exp, conf.reg_strength,
    )
    # compute both singular values and eigenvalues of hessian
    singular_values = jnp.linalg.svd(
        hessian,
        hermitian=True,  # hermitian + real = symmetric
        compute_uv=False  # only return singular values
    )
    eigenvalues, _ = jnp.linalg.eigh(hessian)
    if plot:
        plt.subplots(1, 2, figsize=(10, 5))
        plt.subplot(121)
        # Discrete singular values
        plt.stem(singular_values, jnp.ones_like(singular_values), label=f"Epoch {epoch}")
        plt.xlabel("Singular Value")
        plt.ylim([0, 1.2])
        plt.legend()
        plt.subplot(122)
        # Discrete eigenvalues values
        plt.stem(eigenvalues, jnp.ones_like(eigenvalues), label=f"Epoch {epoch}")
        plt.xlabel("Eigenvalue")
        plt.ylim([0, 1.2])
        plt.legend()
        plt.savefig(
            plots_dir /
            conf.dynamical_model /
            f"sv+eig_plot_{conf.curriculum_learning}_{N_EPOCHS}_{MAX_ITER_OPTX}_{epoch}.pdf"
        )
        plt.show()
    else:
        return eigenvalues, singular_values


def max_grad_norm(grads: PyTree) -> Array:
    # Flatten
    grads_flat, _ = jax.tree_util.tree_flatten(grads)
    # Compute L2 norm
    grad_norms = [jnp.linalg.norm(g) for g in grads_flat]
    # Return max norm
    return jnp.max(jnp.array(grad_norms))
