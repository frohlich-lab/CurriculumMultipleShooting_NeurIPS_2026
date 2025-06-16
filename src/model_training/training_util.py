import diffrax
import equinox as eqx
import git
import itertools as itt
import jax.numpy as jnp
import jax.random as jr
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import pysindy as ps
import seaborn as sns
import wandb
from diffrax import SaveAt

from jaxtyping import Array, PyTree
from pathlib import Path
from scipy.stats import qmc
from src.common import Conf, dynamic_param_labels, hessian_metrics_wandb, plots_dir
from src.model_training.model_util import bounded_tanh_transform_params, set_weight_bias
from src.ude_problem.ude_problem import UDEProblem
from typing import Any, Callable, List, Union


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
    if (conf.opt_steps > 0) and (conf.opt_mult > 1):
        if conf.opt_sched_type != "cos":
            raise ValueError(
                f"Multiple cosine annealing schedules not supported with schedule type: {conf.opt_sched_type}!"
            )
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
    elif (conf.opt_steps == 0) and (conf.opt_mult == 0):
        if conf.opt_sched_type == "exp":
            # Single schedule with warmup and exponential decay
            schedule = {
                "init_value": conf.max_lrate / conf.lrate_span,  # start of annealing
                "peak_value": conf.max_lrate,  # end of annealing
                "warmup_steps": int(n_epochs * conf.warmup_fct),
                "transition_begin": 0,
                "transition_steps": 1,  # rather than decay_rate ** n_epochs, decay_rate ** ((count - transition_begin) / transition_steps)
                "decay_rate": conf.lrate_decay,
                "end_value": conf.max_lrate * conf.lrate_decay ** n_epochs,  # lower bound when to stop
            }
            # plt.loglog(
            #     range(n_epochs),
            #     [optax.schedules.warmup_exponential_decay_schedule(**schedule)(i) for i in range(n_epochs)]
            # )
            # plt.show()
            return optax.schedules.warmup_exponential_decay_schedule(**schedule)
        elif conf.opt_sched_type == "lin":  # Single schedule with linear warmup and decay
            if conf.lrate_decay == 1 and conf.lrate_span == 1 and conf.warmup_fct == 0:
                return optax.constant_schedule(conf.max_lrate)
            # In this case, use conf.lrate_decay as a divider of the initial value to determine the end learning rate
            if conf.lrate_decay < 1:
                decay_factor = 1e2  # default to decaying to 1/100 of the starting value (pre-warmup)
            else:
                decay_factor = conf.lrate_decay

            init_value = conf.max_lrate / conf.lrate_span
            final_value = init_value / decay_factor

            linear_schedule = {
                "transition_steps": n_epochs,
                "peak_value": conf.max_lrate,
                "pct_start": conf.warmup_fct,
                "pct_final": conf.warmup_fct + ((init_value - conf.max_lrate) / (final_value - conf.max_lrate)) * (
                            1.0 - conf.warmup_fct),
                "div_factor": conf.lrate_span,
                "final_div_factor": decay_factor,
            }
            # plt.plot(
            #     range(n_epochs),
            #     [optax.schedules.linear_onecycle_schedule(**linear_schedule)(i) for i in range(n_epochs)]
            # )
            # plt.show()
            return optax.schedules.linear_onecycle_schedule(**linear_schedule)
        else:  # default: cosine annealing
            schedule = {
                "init_value": conf.max_lrate / conf.lrate_span,
                "peak_value": conf.max_lrate,
                "warmup_steps": int(n_epochs * conf.warmup_fct),
                "decay_steps": n_epochs,
                "end_value": conf.max_lrate * conf.lrate_decay ** n_epochs,
            }
            # plt.plot(
            #     range(n_epochs),
            #     [optax.sgdr_schedule([schedule])(i) for i in range(n_epochs)]
            # )
            # plt.show()
            return optax.sgdr_schedule([schedule])
    else:
        raise ValueError(
            f"Invalid configuration for LR schedule: opt_steps={conf.opt_steps}, opt_mult={conf.opt_mult}!"
        )


def get_opt_opt_state_params(
        conf: Conf,
        params: Union[PyTree, UDEProblem],
        schedule: optax.Schedule
):
    # Optimisers to choose from
    optimisers = {
        "adam": optax.adam(schedule),
        "adamw": optax.adamw(schedule),
        "adagrad": optax.adagrad(schedule),
        "sgd": optax.sgd(schedule),
    }

    if conf.optimiser not in optimisers.keys():
        raise ValueError(f"Unknown optimiser: {conf.optimiser}!")

    opt = optimisers[conf.optimiser]
    opt_state = opt.init(params)

    return opt, opt_state, params


def initialise_wandb(
        conf: Conf,
        model: Union[UDEProblem, eqx.nn.MLP],
):
    # Link to repo
    repo = git.Repo(search_parent_directories=True)
    # wandb.require("legacy-service")  # new wandb core
    # Initialise run
    wandb.init(
        project=f"TestUDE.snakemake.{conf.dynamical_model}.{conf.experiment_name}",
        group=f"{conf.training_setting}.{conf.shooting_mode}.{conf.curriculum_learning}",
        config=conf.__dict__,
        name='__'.join(field for field in conf.__str__().split('__')[1:]),  # dynamical_system info already in project
        settings=wandb.Settings(
            start_method="fork",
            git_commit=repo.head.object.hexsha,
            git_remote_url=repo.remotes.origin.url,
        ),
        mode="online"
    )

    # Define shared metrics
    metrics = {
        "loss_train": "min",
        "loss_val": "min",
        "learning_rate": None,
        "patience_counter": None,
    }

    # Add optional Hessian metrics -- both UDE and MLP
    if conf.compute_hessian_metrics:
        metrics = {**metrics, **hessian_metrics_wandb}
    # Add regularisation term if applied -- both UDE and MLP
    if conf.mlp_reg_strength > 0:
        metrics["regularisation_loss"] = "min"

    # Add UDE-only metrics
    if conf.training_setting == "UDE":
        metrics["max_grad_norm"] = "min"
        metrics["num_train_samples"] = None

        # neural network contribution/output regularisation
        if conf.output_reg > 0:
            metrics["output_regularisation_loss"] = "min"

        # End of training metrics (ADAM/BFGS)
        final_metrics = []
        for opt_num_epochs, opt_label in zip([conf.num_epochs_adam, conf.num_epochs_bfgs], ["ADAM", "BFGS"]):
            if opt_num_epochs > 0:
                final_metrics.extend(
                    [
                        f"final_loss_{dataset}_{opt_label}" for dataset in ["train", "val"]
                    ] +
                    [
                        f"final_{param_label}_{opt_label}"
                        for param_label in dynamic_param_labels[conf.dynamical_model]
                    ]
                )
        metrics = {
            **metrics,
            **{
                final_metric: None
                for final_metric in final_metrics
            }
        }


    # Define metrics in W&B
    for metric in metrics.keys():
        # if metric summary not specified
        if metrics[metric] is None:
            wandb.define_metric(metric)
        else:
            wandb.define_metric(metric, summary=metrics[metric])

    if isinstance(model, UDEProblem):
        mlp = model.func.mlp
    elif isinstance(model, eqx.nn.MLP):
        mlp = model
    else:
        raise ValueError(f"Unknown model type: {type(model)}!")
    # Define layer statistics (weight/bias value and grads)
    # from Fabian's NeuralCoarseGraining
    # https://github.com/frohlich-lab/NeuralCoarseGraining/blob/8c5cc42447b9b55cdf8b3bfc685388ba4bf6b751/ncg/training.py#L257
    for il, _ in enumerate(mlp.layers):
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
        frozen_mechanistic: bool = False,
        frozen_nn: bool = False,
):
    """
    Log the layer statistics to wandb.

    :param model: model containing weight information
    :param grad: model containing gradient information
    :param conf: configuration object
    :param frozen_mechanistic: bool, whether the mechanistic parameters are frozen
    :param frozen_nn: bool, whether the neural network is frozen
    """
    layers = [
        (par_layer, grad_layer)
        for par_layer, grad_layer in zip(
            model.func.mlp.layers, grad.func.mlp.layers, strict=False
        )
        if isinstance(par_layer, eqx.nn.Linear)
    ]
    # TODO @GiacomoFabrini - would be good to streamline the following code
    if not frozen_nn:
        if conf.use_bias:
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
        else:
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
                    ('w', ),
                    (layer.weight.ravel(), ),
                    strict=False,
                )
            }
    else:
        if conf.use_bias:
            layer_stats = {
                f'layer{ind_layer}.{par_label}_{value_label}': wandb.Histogram(list(par_vals))
                for ind_layer, (par_layer, _) in enumerate(layers)
                for value_label, layer in zip(['vals'], [par_layer])
                for par_label, par_vals in zip(
                    ('w', 'b'),
                    (layer.weight.ravel(), layer.bias.ravel()),
                    strict=False,
                )
            }
        else:
            layer_stats = {
                f'layer{ind_layer}.{par_label}_{value_label}': wandb.Histogram(list(par_vals))
                for ind_layer, (par_layer, _) in enumerate(layers)
                for value_label, layer in zip(['vals'], [par_layer])
                for par_label, par_vals in zip(
                    ('w', ),
                    (layer.weight.ravel(), ),
                    strict=False,
                )
            }

    # Define parameter labels
    param_labels = dynamic_param_labels[conf.dynamical_model]

    if not frozen_mechanistic:
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
                ('vals',),
                (model.func.dynamic_parameters,),
                strict=False
            )
            for par_label, par_val in zip(
                param_labels,
                np.array(par_vals).ravel(),
                strict=False
            )
        }

    # Log values and grads for shooting window initial values
    initial_values_labels = [
        "u0_" + str(i+1) for i in range(model.initial_values.shape[0])
    ]
    initial_value_stats = {
        f'initial_values.{par_label}_{value_label}': wandb.Histogram(
            np.log10(np.abs(np.array(par_val[par_val != 0])))
        )
        if value_label == 'grads'
        else wandb.Histogram(par_val)
        for value_label, par_vals in zip(
            ('vals', 'grads'),
            (model.initial_values, grad.initial_values),
            strict=False
        )
        for par_label, par_val in zip(
            initial_values_labels,
            np.array(par_vals).ravel(),
            strict=False
        )
    }

    stats = {**layer_stats, **dynamic_param_stats, **initial_value_stats}
    return stats


def log_plots(
        df_predict_list: List[pd.DataFrame],
        df_obs: pd.DataFrame,
        dataset: str,
        shooting_mode: str,
        curriculum_learning: bool,
        colors: List[str],
        epoch: int,
        plot_name: str,
        plot_dir: Path,
        save_plot: bool = False,
        plot_fmt: str = "png"
):
    df_predict, df_predict_shooting_train = df_predict_list
    plt.subplots(1, 2, figsize=(6, 4), sharey=True, sharex=True)
    for ind, (label, color) in enumerate(
            zip([col for col in df_obs.columns if col != "time"], colors)
    ):
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
        # For multiple-shooting (MS, without CL), plot simulations and delimiters for the individual shooting windows
        if (shooting_mode == "multiple") and (dataset == "train") and not curriculum_learning:
            for window in sorted(df_predict_shooting_train.window.unique()):
                # Only draw line between points corresponding to the same window
                plt.plot(
                    df_predict_shooting_train[
                        df_predict_shooting_train.window == window
                    ]["time"].values,
                    df_predict_shooting_train[
                        df_predict_shooting_train.window == window
                    ][label].values,
                    color=color,
                    ls='--',
                    alpha=0.5,
                )
                plt.axvline(
                    x=df_predict_shooting_train[
                        df_predict_shooting_train.window == window
                    ]["time"].values[0],
                    color='black',
                    alpha=0.6,
                    ls='dotted'
                )

        plt.legend(frameon=False)
        plt.xlabel('Time')
        plt.gca().tick_params(direction='in', length=5)
        if ind == 0:
            plt.ylabel('Measurement')

    if save_plot and (plot_dir is not None):
        plt.savefig(plot_dir / f"fit_plot_{plot_name}.{plot_fmt}")
        plt.close()
    elif save_plot and (plot_dir is None):
        plt.show()
        raise ValueError("Must specify plot directory in order to save plots!")
    else:
        wandb.log({f"fit_plot_{plot_name}": plt}, step=epoch)
        plt.close()


def make_grads_finite(grads):
    return jax.tree_map(
        lambda x: jnp.where(jnp.isfinite(x), x, jnp.zeros_like(x)),
        grads,
    )


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
        data: Array,
        t_save: Union[diffrax.SaveAt, Array],
        t_span: List,
        n_epoch: int,
        curriculum_params: dict[int, Any]
):
    # Initialize the list to store the results
    incremental_data = []
    incremental_t_save = []
    incremental_t_span = []
    incremental_steps = [curriculum_params[stage]["start"] for stage in sorted(curriculum_params.keys())]

    # Fill the incremental data list with the incremental data
    for step in sorted(curriculum_params.keys()):
        length = curriculum_params[step]["length"]
        start_epoch = curriculum_params[step]["start"]
        end_epoch = curriculum_params[step]["end"]
        for _ in range(end_epoch - start_epoch):
            incremental_data.append(data[:length])
            incremental_t_save.append(diffrax.SaveAt(ts=t_save.subs.ts[:length]))
            incremental_t_span.append([np.min(t_save.subs.ts[:length]), np.max(t_save.subs.ts[:length])])


    # Add full data for the remaining epochs
    epochs_filled = len(incremental_data)

    for _ in range(n_epoch - epochs_filled):
        incremental_data.append(data)
        incremental_t_save.append(t_save)
        incremental_t_span.append(t_span)

    return incremental_data, incremental_t_save, incremental_t_span, incremental_steps


def max_grad_norm(grads: PyTree) -> Array:
    # Flatten
    grads_flat, _ = jax.tree_util.tree_flatten(grads)
    # Compute L2 norm
    grad_norms = [jnp.linalg.norm(g) for g in grads_flat]
    # Return max norm
    return jnp.max(jnp.array(grad_norms))


def sample_dynamical_params(true_params, deviation, seed, scale="log10", method="sample_uniform", **kwargs):
    # Perturbation/sampling is always performed in log10-space
    if method == "lh":  # lh=latin_hypercube
        sampler = qmc.LatinHypercube(d=true_params.shape[0], seed=seed)
        sampled_deviation = 2*(sampler.random(n=1).flatten() - 0.5)  # shift from [0, 1) to [-1, 1) range
        sampled_deviation *= deviation
        starting_params_log10 = jnp.log10(true_params) + sampled_deviation
    elif method == "su": # su=sample_uniform
        starting_params_log10 = jax.random.uniform(
            key=jr.PRNGKey(seed),
            shape=true_params.shape,
            minval=jnp.log10(true_params)-deviation,
            maxval=jnp.log10(true_params)+deviation,
        )
    elif method == "agn":  # agn=add_gaussian_noise, perturbs original values
        log_true_params = jnp.log10(true_params)
        starting_params_log10 = log_true_params + deviation*jax.random.normal(
            key=jr.PRNGKey(seed),
            shape=true_params.shape,
        )
    else:
        raise ValueError(
            f"Found unknown method {method}. Method must be either 'sample_uniform' or 'add_gaussian_noise'."
        )
    if scale == "log10":
        return starting_params_log10
    elif scale == "linear":
        return jnp.power(10, starting_params_log10)
    elif scale == "bounded_tanh":
        return bounded_tanh_transform_params(
            params=jnp.power(10, starting_params_log10),
            **kwargs, # pass additional arguments to the bounded_tanh_transform_params function (lower/upper bounds, loc)
        )
    else:
        raise ValueError(f"Unknown parameter estimation scale: {scale}!")

def enforce_minimum_spacing(arr: np.ndarray, min_dist: int) -> np.ndarray:
    """
    Filters an input array, arr, and keeps the original items that are at least min_dist apart.
    """
    prev = - np.inf  # ensure first element is always kept
    return np.array([prev := x for x in arr if x - prev >= min_dist])


def generate_log_epochs(n_epochs: int, num_samples: int, min_dist: int) -> np.ndarray:
    """
    Returns epochs regularly spaced in log10 space but no closer than min_dist
    to prevent running into filestream rate limit in W&B.
    """
    log_epochs = np.unique(
        np.logspace(0, np.log10(n_epochs), num=num_samples).astype(int)
    )
    return enforce_minimum_spacing(log_epochs, min_dist)


def get_hessian_metrics(hessian: Array, return_eigs_svs: bool=False) -> Union[tuple[dict, Array, Array], dict]:
    hessian_trace = jnp.trace(hessian)
    hessian_diag = hessian * jnp.eye(hessian.shape[0])
    max_abs_diag = jnp.max(jnp.abs(hessian_diag))
    frob_diag = jnp.linalg.norm(hessian_diag)
    frob_off_diag = jnp.linalg.norm(hessian - hessian_diag)
    max_abs_off_diag = jnp.max(jnp.abs(hessian - hessian_diag))
    frob_ratio_off_to_diag = frob_off_diag / frob_diag

    # Compute singular values and eigenvalues
    singular_values = jnp.linalg.svd(
        hessian,
        hermitian=True,  # hermitian + real = symmetric
        compute_uv=False  # only return singular values
    )
    eigenvalues, _ = jnp.linalg.eigh(hessian)
    hessian_metrics = {
        "hessian_trace": hessian_trace,
        "max_abs_diagonal_element": max_abs_diag,
        "max_abs_off_diagonal_element": max_abs_off_diag,
        "frobenius_norm_diagonal": frob_diag,
        "frobenius_norm_off_diagonal": frob_off_diag,
        "ratio_frobenius_norm_offdiag_to_diag": frob_ratio_off_to_diag,
        "max_eigenvalue": jnp.max(eigenvalues),
        "min_eigenvalue": jnp.min(eigenvalues),
        "max_sv": jnp.max(singular_values),
        "min_sv": jnp.min(singular_values),
        "condition_number": jnp.max(singular_values)/jnp.min(singular_values),
    }
    return (hessian_metrics, eigenvalues, singular_values) if return_eigs_svs else hessian_metrics


# Tested (tests/test_training.py)
class MetricHandler:
    def __init__(self, patience=5):
        self.counter = 0
        self.patience = patience
        self.should_break = False

    def check_whether_should_break(self, metric):
        if not jnp.isfinite(metric):
            self.counter += 1
            if self.counter >= self.patience:  # fixed budget of patience
                print("Too many invalid values, breaking now.")
                self.should_break = True
        else:
            self.counter = 0
        return self.should_break


def transform_dynamic_params(dynamic_parameters: jnp.ndarray, conf: Conf) -> jnp.ndarray:
    if conf.dynamic_params_scale == "log10":
        return jnp.power(10, dynamic_parameters)
    elif conf.dynamic_params_scale == "linear":
        return dynamic_parameters
    else:
        raise ValueError(f"Unknown dynamic parameter scale: {conf.dynamic_params_scale}!")


def get_ude_trajectory(
        model: UDEProblem,
        u0,
        t_save,
        t_span,
        mlp_only: bool = False,
        return_full_solution: bool = False
):
    solver = diffrax.Tsit5()

    if len(u0) == model.func.state_dimension and model.func.regularise_network:
        u0 = jnp.concatenate([u0, jnp.zeros(1)])

    if mlp_only:
        # Zero out dynamical parameters -- only use MLP contribution to determine dynamics
        model = model.replace_dynamic_parameters(
            new_dynamic_parameters=jnp.zeros(len(model.func.dynamic_parameters))
        )
        # Set dynamic parameter scale to linear
        model = model.replace_options(
            new_options={"dynamic_params_scale": "linear"}
        )

    # Compute solution using single-shooting
    solution = model.solve_single_shooting_validation(
        solver,
        u0,
        t_save,
        t_span,
    )

    return solution if return_full_solution else solution.ys[:, :model.func.state_dimension]


def get_ude_trajectory_and_derivatives(model: UDEProblem, u0, t_save, t_span, mlp_only: bool = False):
    # Save dense solution
    t_save = diffrax.SaveAt(ts=t_save.subs.ts, dense=True)
    trajectory_solution = get_ude_trajectory(model, u0, t_save, t_span, mlp_only, return_full_solution=True)
    # Evaluate the vector field to get true derivatives
    vector_field = model.func
    derivatives = np.array(
        [
            vector_field(t, trajectory_solution.evaluate(t), args=u0)
            for t in t_save.subs.ts
        ]
    )
    return dict(
        traj=trajectory_solution.ys[:, :model.func.state_dimension],
        der=derivatives[:, :model.func.state_dimension]
    )


def get_sindy_formula(
        model: UDEProblem,
        u0: jnp.ndarray,
        t_save: diffrax.SaveAt,
        t_span: List[float],
        settings: List[str],
        use_derivatives: bool
):
    # Get trajectories + true derivatives (vector field)
    traj_der = {
        tag: get_ude_trajectory_and_derivatives(
            model, u0, t_save, t_span, mlp_only=zero_mlp
        )
        for tag, zero_mlp in zip(["MLP", "UDE"], [True, False])
    }
    ps_models = {}
    for setting in settings:  # missing physics & full model (known + missing)
        ps_models[setting] = ps.SINDy(
            feature_names=[f"x{i + 1}" for i in range(model.func.state_dimension)]
        )
        if not use_derivatives:
            ps_models[setting].fit(
                x=traj_der[setting]["traj"],
                t=np.array(t_save.subs.ts)
            )  # numerically estimate derivatives
        else:
            ps_models[setting].fit(
                x=traj_der[setting]["traj"],
                t=np.array(t_save.subs.ts),
                x_dot=traj_der[setting]["der"],  # use derivatives from vector field evaluations
            )
        # Print found symbolic regression model
        ps_models[setting].print()
        print(
            f"Number of non-zero coefficients in SINDy model of {setting}: "
            f"{(np.abs(ps_models[setting].coefficients()) > 0).sum()}"
        )


def generate_epochs_dict(epochs_list):
    # Validate that the list has an even number of elements
    if len(epochs_list) % 2 != 0:
        raise ValueError("The epochs list must have an even number of elements.")

    # Create the dictionary
    epochs_dict = {}
    num_steps = len(epochs_list) // 2

    for i in range(num_steps):
        start_epoch = epochs_list[2 * i]
        end_epoch = epochs_list[2 * i + 1]
        epochs_dict[i] = {"start": start_epoch, "end": end_epoch}
    return epochs_dict


def augment_u0(u0):
    """
    Augment the initial condition(s) with additional zero(es) element to compute
    network contribution regularisation.
    """
    # Dynamically compute the shape for the new zeros array
    zeros_shape = u0.shape[:-1] + (1,)
    # Concatenate along the last axis
    return jnp.concatenate([u0, jnp.zeros(zeros_shape)], axis=-1)


def prep_bfgs_training(model, zero_out_mlp, use_output_reg: bool):
    # Zero out MLP if needed (mechanistic fit only)
    if zero_out_mlp:
        mlp_replacement = set_weight_bias(model.func.mlp, 0.0, jax.random.PRNGKey(0))
    else:
        mlp_replacement = model.func.mlp

    # Set dropout behaviour to be deterministic and enable output/neural network contribution regularisation
    model_bfgs = model.replace_mlp(
        new_mlp=mlp_replacement,
        switch_to_inference=True,
        regularise_network=use_output_reg if (use_output_reg and not model.func.regularise_network) else None
    )
    return model_bfgs


def get_sol_and_loss_validation(
        prob: UDEProblem,
        u0: Array,
        t_save: SaveAt,
        t_span: List,
        adjoint: diffrax.AbstractAdjoint,
        loss_fn: Callable,
        data: Array,
        max_steps: int = 40000  # 2x as long for 2x timecourse
):
    sol = prob.solve_single_shooting_validation(
        diffrax.Tsit5(),
        u0,
        t_save,  # use provided t_save
        t_span,  # use provided t_span
        adjoint=adjoint,
        max_steps=max_steps
    )
    loss = loss_fn(
        solution=sol.ys,
        data=data,
    )
    return sol, loss


def get_sol_and_loss_single_shooting(
        prob: UDEProblem,
        adjoint: diffrax.AbstractAdjoint,
        loss_fn: Callable,
        data: Array,
        t_save_windows: Array,
        masks: Array,
        max_steps: int = 20000
):
    sol = prob.solve_single_shooting(
        diffrax.Tsit5(),
        prob.initial_values[0],  # pass 1D array
        t_save_windows[0],
        adjoint,
        max_steps
    )
    loss = loss_fn(
        solution=sol.ys,
        data=data,
        mask=masks.squeeze()
    )
    return sol, loss


def get_sol_and_loss_multiple_shooting(
        prob: UDEProblem,
        adjoint: diffrax.AbstractAdjoint,
        loss_fn: Callable,
        data: Array,
        t_save_windows: Array,
        masks: Array,
        last_valid_index: int,
        max_steps: int = 20000
):
    sol = prob.solve_multiple_shooting(
        diffrax.Tsit5(),
        prob.initial_values[:last_valid_index],
        t_save_windows[:last_valid_index],
        adjoint,
        max_steps
    )
    losses = jnp.array([
        loss_fn(
            solution=sol.ys[i, :, :prob.func.state_dimension],
            data=data,
            mask=masks[i]
        )
        for i in range(last_valid_index)
    ])
    return sol, jnp.mean(losses)


def check_and_progress_curriculum(
        t_save_windows,
        masks,
        nn_epoch: int,
        cl_stage: int,
        model_save_steps_start: List,
        curriculum_params: dict
):
    advancing_cl_stage = False
    # If the epoch matches, update CL stage, inner timepoints and masks, and zero last set of initial values (window)
    if (nn_epoch != 0) and (nn_epoch in model_save_steps_start):
        print("----------------------------------")
        cl_stage = int(
            jnp.argmin(
                jnp.abs(jnp.array(model_save_steps_start) - nn_epoch)
            )
        ) + 1  # 1-counting
        t_save_windows = curriculum_params[cl_stage]["time"]
        masks = curriculum_params[cl_stage]["mask"]
        advancing_cl_stage = True
    return t_save_windows, masks, cl_stage, advancing_cl_stage


def get_save_epochs(num_epochs, num_models):
    if num_models <= 0 or num_epochs <= 0:
        return []
    if num_models == 1:
        return [num_epochs - 1]

    return [round(i * (num_epochs - 1) / (num_models - 1)) for i in range(num_models)]
