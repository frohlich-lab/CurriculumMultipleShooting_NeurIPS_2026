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

from flax.training.early_stopping import EarlyStopping
from jaxtyping import Array, PyTree
from pathlib import Path
from src.common import Conf, dynamic_param_labels, EarlyStoppingParams, hessian_metrics_wandb, plots_dir
from src.model_training.model_util import bounded_tanh_transform_params
from src.model_training.training_configuration import N_EPOCHS, MAX_ITER_OPTX_FINETUNE
from src.ude_problem import UDEProblem
from typing import List, Optional, Union


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
        "adagrad": optax.adagrad(schedule),
        # "adam_lbfgs": optax.chain(
        #     optax.adam(schedule),
        #     optax.scale_by_lbfgs()
        # ),
        "sgd": optax.sgd(schedule),
        # "sgd_lbfgs": optax.chain(
        #    optax.sgd(schedule),
        #    optax.scale_by_lbfgs()
        # )
    }

    if conf.optimiser not in optimisers.keys():
        raise ValueError(f"Unknown optimiser: {conf.optimiser}!")

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
        model: Union[UDEProblem, eqx.nn.MLP],
):
    # Link to repo
    repo = git.Repo(search_parent_directories=True)
    # Require legacy back-end (for now, until we understand if the new core works with NEMO) TODO @GiacomoFabrini revisit this!
    wandb.require("legacy-service")
    # Initialise run
    wandb.init(
        project=f"TestUDE.nf.{conf.dynamical_model}.{conf.experiment_name}",  # nf = nextflow pipeline
        group=f"{conf.dynamical_model}.{conf.training_setting}",
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
            # "frozen_mechanistic_params_optx"
            # if conf.freeze_mechanistic_params_optx else "learnable_mechanistic_params_optx",  # DISABLED TO MERGE WITH MLP
        ],
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

        # Add LV BFGS local minima experiment-specific metrics (loss on entire training set at each curriculum learning step)
        if conf.dynamical_model == "lotka_volterra":
            if conf.curriculum_loss_landscape_BFGS:
                metrics["full_train_loss"] = "min"

        # End of training metrics (ADAM/BFGS)
        final_metrics = []
        for opt_num_epochs, opt_label in zip([N_EPOCHS, MAX_ITER_OPTX_FINETUNE], ["ADAM", "BFGS"]):
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
        t_save: diffrax.SaveAt,
        n_epoch: int,
        curriculum_learning: bool,
        num_curriculum_steps: int,
        max_curriculum_epochs: Optional[int],
):
    # Initialize the list to store the results
    incremental_data = []
    incremental_t_save = []
    incremental_t_span = []

    # Initialise step list
    incremental_steps = []

    # Initialise variables
    epochs_filled = 0

    if curriculum_learning:
        # Check number of curriculum steps
        if num_curriculum_steps is None:
            raise ValueError("Number of steps must be defined for curriculum learning!")
        elif num_curriculum_steps <= 0:
            raise ValueError("Number of steps must be positive!")

        # Check number of epochs to spend on curriculum learning
        if max_curriculum_epochs is None:
            raise ValueError("Max curriculum steps must be defined for curriculum learning!")
        elif max_curriculum_epochs <= 0 or max_curriculum_epochs > n_epoch:
            raise ValueError("Max curriculum steps cannot be negative!")
        # Calculate the epoch at which to switch to full data
        switch_epoch = min(max_curriculum_epochs, n_epoch)

        # Fill the incremental data list with the incremental data
        for step in range(1, num_curriculum_steps + 1):
            length = int(len(data) * (step / num_curriculum_steps))
            start_epoch = int((step - 1) * (switch_epoch / num_curriculum_steps))
            end_epoch = int(step * (switch_epoch / num_curriculum_steps))
            incremental_steps.append(start_epoch)
            for _ in range(end_epoch - start_epoch):
                incremental_data.append(data[:length])
                incremental_t_save.append(diffrax.SaveAt(ts=t_save.subs.ts[:length]))
                incremental_t_span.append([np.min(t_save.subs.ts[:length]), np.max(t_save.subs.ts[:length])])

        epochs_filled = len(incremental_data)

    # Add full data for the remaining epochs
    for _ in range(n_epoch - epochs_filled):
        incremental_data.append(data)
        incremental_t_save.append(t_save)
        incremental_t_span.append([np.min(t_save.subs.ts), np.max(t_save.subs.ts)])

    return incremental_data, incremental_t_save, incremental_t_span, incremental_steps


def eig_sv_plot(
        eigenvalues: jnp.ndarray,
        singular_values: jnp.ndarray,
        conf: Conf,
        epoch: int,
):
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
        f"sv+eig_plot_{conf.curriculum_learning}_{N_EPOCHS}_{MAX_ITER_OPTX_FINETUNE}_{epoch}.pdf"
    )
    plt.show()


def max_grad_norm(grads: PyTree) -> Array:
    # Flatten
    grads_flat, _ = jax.tree_util.tree_flatten(grads)
    # Compute L2 norm
    grad_norms = [jnp.linalg.norm(g) for g in grads_flat]
    # Return max norm
    return jnp.max(jnp.array(grad_norms))


def sample_dynamical_params(true_params, deviation, seed, scale="log10", method="sample_uniform", **kwargs):
    # Perturbation/sampling is always performed in log10-space
    if method == "sample_uniform":
        starting_params_log10 = jax.random.uniform(
            key=jr.PRNGKey(seed),
            shape=true_params.shape,
            minval=jnp.log10(true_params)-deviation,
            maxval=jnp.log10(true_params)+deviation,
        )
    elif method == "add_gaussian_noise":  # add option to perturb parameters with gaussian noise rather than sample from uniform
        log_true_params = jnp.log10(true_params)
        starting_params_log10 = log_true_params + deviation*jax.random.normal(
            key=jr.PRNGKey(seed),
            shape=true_params.shape,
        )
    else:
        raise ValueError(
            f"Found uknown method {method}. Method must be either 'sample_uniform' or 'add_gaussian_noise'."
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
        model = model.copy(
            new_dynamic_parameters=jnp.zeros(len(model.func.dynamic_parameters)),
            new_options={"dynamic_params_scale": "linear"}
        )

    # Compute solution
    solution = model.solve(
        solver,
        u0,
        t_save,
        t_span
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

def configure_early_stopping(early_stopping_params):
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
    else:
        early_stopper = None
    return early_stopper