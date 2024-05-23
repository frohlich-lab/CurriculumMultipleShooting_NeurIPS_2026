import dataclasses
import equinox as eqx
import git
import itertools as itt
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import optimistix as optx
import pandas as pd
import wandb

from src.common import Conf, EarlyStoppingParams, plots_dir
from flax.training.early_stopping import EarlyStopping
from jaxtyping import Array, Float, PyTree
from pathlib import Path
from src.model_examples.lotka_volterra.setup_diffrax_model import UDEProblem as UDEProblem_lv
from src.model_examples.lotka_volterra.setup_diffrax_model import get_fit_dfs as get_fit_dfs_lv
from src.model_examples.sir.setup_diffrax_model import UDEProblem as UDEProblem_sir
from src.model_examples.sir.setup_diffrax_model import get_fit_dfs as get_fit_dfs_sir
from typing import Callable, Dict, List

# Optimisers to choose from
optimisers = {
    "adam": optax.adam,
    "adamw": optax.adamw,
}


def train(
        model: UDEProblem_lv | UDEProblem_sir,
        u0: jnp.ndarray,
        training_data: jnp.ndarray,
        t_save_train: List,
        t_span_train: List,
        validation_data: jnp.ndarray,
        t_save_val: List,
        t_span_val: List,
        conf: Conf,
        n_epochs: int,
        early_stopping_params: EarlyStoppingParams,
        loss_fn: Callable,
        loss_grad_fn: Callable,
        loss_optx: Callable,
        optx_config: Dict,
        max_iter_optx: int,
) -> UDEProblem_lv | UDEProblem_sir:
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
    :param n_epochs:
        Number of training iterations (epochs).
    :param early_stopping_params:
        Dictionary of parameters for Early-Stopping.
    :param loss_fn:
        Loss function to use for training.
    :param loss_grad_fn:
        Loss function to use for training which also enables gradient computation.
    :param loss_optx:
        Loss function to be used with optimistix for further training (only takes learnable params,
        i.e. dynamic portion of the model).
    :param optx_config:
        Configuration for optimistix solver (rtol, atol).
    :param max_iter_optx:
        Maximum number of iterations for further training via optimistix.
    """

    # Initialise wandb and define metrics
    initialise_wandb(
        conf=conf,
        early_stopping_params=early_stopping_params,
        model=model,
    )

    if not conf.use_lookahead:
        # Get schedule and initialise the optimiser
        schedule = get_scheduler(conf=conf, n_epochs=n_epochs)
        opt = optimisers[conf.optimiser](schedule)
        opt_state = opt.init(eqx.filter(model, eqx.is_array))
    else:
        opt = optax.lookahead(
            fast_optimizer=optimisers[conf.optimiser](conf.max_lrate),
            sync_period=conf.lookahead_sync,
            slow_step_size=conf.lookahead_step_size,
            reset_state=conf.lookahead_reset_state,
        )
        params = optax.LookaheadParams.init_synced(eqx.filter(model, eqx.is_array))
        opt_state = opt.init(params)

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
            model: UDEProblem_lv | UDEProblem_sir,
            opt_state: PyTree,
            input_data: Float[Array, '...'],  # TODO @GiacomoFabrini fix input data shape
            t_save: List,
            t_span: List,
            conf: Conf,
    ):
        loss_value, grads = loss_grad_fn(
            model,
            u0,
            t_save,
            t_span,
            input_data,
            conf.reg_exp,
            conf.reg_strength,
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
            conf=conf,
        )
        if conf.use_dropout:
            # Get a copy of the model for inference
            inference_model = model.copy(
                new_mlp=eqx.nn.inference_mode(model.func.mlp, value=True)
            )
        else:
            # no need for difference in training/inference behaviour
            inference_model = model
        # compute validation loss
        loss_val = loss_fn(
            inference_model,
            u0,
            t_save_val,
            t_span_val,
            validation_data,
            conf.reg_exp,
            conf.reg_strength,
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
                    **log_model_stats(model, grads),
                },
                step=epoch
            )

        # Check early stopping criterion
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

    # Plot fit (train/val) at end of training
    if conf.dynamical_model == "lotka_volterra":
        get_fit_dfs = get_fit_dfs_lv
        labels = ['prey', 'predator']
        colors = ['red', 'black']
    elif conf.dynamical_model == 'sir':
        get_fit_dfs = get_fit_dfs_sir
        labels = ['Infected', 'Recovered + dead']
        colors = ['red', 'black']
    else:
        raise ValueError(f"Undefined dynamical model {conf.dynamical_model}.")
    for dataset, t_save, t_span, data in zip(
            ["train", "validation"],
            [t_save_train, t_save_val],
            [t_span_train, t_span_val],
            [training_data, validation_data]
    ):
        df_predict, df_obs = get_fit_dfs(model, u0, t_save, t_span, data)
        plot_dir = plots_dir / conf.dynamical_model / "optax"
        plot_dir.mkdir(exist_ok=True, parents=True)
        log_plots(
            df_predict,
            df_obs,
            labels,
            colors,
            epoch,
            f"{dataset}",
            save_plot=False,
            plot_dir=plot_dir,
            plot_fmt="svg"
        )

    wandb.log({"final_epoch": epoch})

    if max_iter_optx > 0:  # otherwise end training and return trained model
        # Additional training with BFGS via optimistix
        solver = optx.BFGS(rtol=optx_config["rtol"], atol=optx_config["atol"])
        # Set dropout behaviour to be deterministic
        model_optx = model.copy(
                new_mlp=eqx.nn.inference_mode(model.func.mlp, value=True)
            )
        # Split model in learnable params and static component
        params, static = eqx.partition(model_optx, eqx.is_array)
        # Minimise via Optimistix
        sol = optx.minimise(
            fn=loss_optx,
            solver=solver,
            y0=params,
            args=(static, u0, t_save_train, t_span_train, training_data),
            max_steps=max_iter_optx,
        )
        # Update model
        params = sol.value
        model = eqx.combine(params, static)
        # Compute final loss_train and loss_val and log them to wandb
        loss_dict = {
            dataset: loss_fn(
                model,
                u0,
                t_save,
                t_span,
                data,
                conf.reg_exp,
                conf.reg_strength,
            )
            for t_save, t_span, data, dataset in zip(
                [t_save_train, t_save_val],
                [t_span_train, t_span_val],
                [training_data, validation_data],
                ["train", "validation"],
            )
        }

        wandb.log(
            {
                "final_loss_train_optx": loss_dict["train"],
                "final_loss_val_optx": loss_dict["validation"],
            },
            step=epoch+sol.stats["num_steps"],
        )

        # Log plots again after further training
        for dataset, t_save, t_span, data in zip(
                ["train", "validation"],
                [t_save_train, t_save_val],
                [t_span_train, t_span_val],
                [training_data, validation_data]
        ):
            df_predict, df_obs = get_fit_dfs(model, u0, t_save, t_span, data)
            plot_dir = plots_dir / conf.dynamical_model / "optx"
            plot_dir.mkdir(exist_ok=True, parents=True)
            log_plots(
                df_predict,
                df_obs,
                labels,
                colors,
                epoch+sol.stats["num_steps"],
                f"{dataset}_optx",
                save_plot=False,
                plot_dir=plot_dir,
                plot_fmt="svg",
            )

    wandb.finish()
    return model


# from Fabian's NeuralCoarseGraining
def log_model_stats(
        model: UDEProblem_lv | UDEProblem_sir,
        grad: UDEProblem_lv | UDEProblem_sir,
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
            ('alpha', 'delta'),
            np.array(par_vals).ravel(),
            strict=False
        )
    }

    stats = {**layer_stats, **dynamic_param_stats}
    return stats


def initialise_wandb(
        conf: Conf,
        early_stopping_params: EarlyStoppingParams,
        model: UDEProblem_lv | UDEProblem_sir,
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
        ]
    )

    # Define some metrics
    metrics = {
        "loss_train": "min",
        "loss_val": "min",
        "learning_rate": None,
        "patience_counter": None,
        "final_loss_train_optx": "min",
        "final_loss_val_optx": "min",
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
    plt.subplots(1, 2, figsize=(6, 4))
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
