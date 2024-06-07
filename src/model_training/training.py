import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import optimistix as optx
import wandb

from src.common import Conf, EarlyStoppingParams, plots_dir
from flax.training.early_stopping import EarlyStopping
from jaxtyping import Array, Float, PyTree
from src.model_examples.lotka_volterra.setup_diffrax_model import get_fit_dfs as get_fit_dfs_lv
from src.model_examples.sir.setup_diffrax_model import get_fit_dfs as get_fit_dfs_sir
from src.model_training.training_util import get_scheduler, initialise_wandb, log_model_stats, log_plots
from src.ude_problem import UDEProblem
from types import SimpleNamespace
from typing import Callable, Dict, List

# Optimisers to choose from
optimisers = {
    "adam": optax.adam,
    "adamw": optax.adamw,
}


def make_grads_finite(grads):
    return jax.tree_map(
        lambda x: jnp.where(jnp.isfinite(x), x, jnp.zeros_like(x)),
        grads,
    )


def optx_solve(y, solver, fn, args, args_val, f_struct, aux_struct, options, tags, max_steps, starting_epoch, conf):
    # These arguments are always fixed throughout interactive solves.
    step = eqx.filter_jit(
        eqx.Partial(solver.step, fn=fn, args=args, options=options, tags=tags)
    )
    terminate = eqx.filter_jit(
        eqx.Partial(solver.terminate, fn=fn, args=args, options=options, tags=tags)
    )

    # Initial state before we start solving.
    state = solver.init(fn, y, args, options, f_struct, aux_struct, tags)
    done, result = terminate(y=y, state=state)

    iteration = 1
    while not (done or iteration > max_steps):
        # print(f"Evaluating point {y} with value {fn(y, args)}.")
        y, state, aux = step(y=y, state=state)
        # Log training and validation loss + params and grads every 5 optimistix iterations
        if iteration % 5 == 0:
            loss_train = fn(y, args)[0]
            loss_val = fn(y, args_val)[0]
            wandb.log(
                {
                    "loss_train": loss_train,
                    "loss_val": loss_val,
                    **log_model_stats(
                        eqx.combine(y, args[0]),  # recombine updated params and static
                        state.f_info.grad,
                        conf
                    ),
                },
                step=starting_epoch+iteration
            )
        iteration += 1
        done, result = terminate(y=y, state=state)
    if result != optx.RESULTS.successful:
        print(f"Oh no! Got error {result}.")
    y, _, _ = solver.postprocess(fn, y, aux, args, options, state, tags, result)
    # print(f"Found solution {y} with value {fn(y, args)}.")
    # Log final training and validation loss terms
    wandb.log(
        {
            "loss_train": loss_train,
            "loss_val": loss_val,
        },
        step=starting_epoch + iteration
    )
    sol = {
        "value": y,
        "stats": {
            "num_steps": iteration,
            "termination_reason": result,
        },
    }
    return SimpleNamespace(**sol)


@eqx.filter_jit
def make_step(
        params: UDEProblem,
        static: UDEProblem,
        opt: optax.GradientTransformation,
        opt_state: PyTree,
        input_data: Float[Array, '...'],  # TODO @GiacomoFabrini fix input data shape
        u0: jnp.ndarray,
        t_save: List,
        t_span: List,
        conf: Conf,
        loss_grad_fn: Callable,
):
    loss_value, grads = loss_grad_fn(
        params,
        static,
        u0,
        t_save,
        t_span,
        input_data,
        conf.reg_exp,
        conf.reg_strength,
    )
    grads = make_grads_finite(grads)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = eqx.apply_updates(params, updates)
    return params, opt_state, loss_value, grads


@eqx.filter_jit
def make_step_looahead(
        params: UDEProblem,
        static: UDEProblem,
        opt: optax.GradientTransformation,
        opt_state: PyTree,
        input_data: Float[Array, '...'],  # TODO @GiacomoFabrini fix input data shape
        u0: jnp.ndarray,
        t_save: List,
        t_span: List,
        conf: Conf,
        loss_grad_fn: Callable,
):
    loss_value, grads = loss_grad_fn(
        params.fast,
        static,
        u0,
        t_save,
        t_span,
        input_data,
        conf.reg_exp,
        conf.reg_strength,
    )
    grads = make_grads_finite(grads)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = eqx.apply_updates(params, updates)
    return params, opt_state, loss_value, grads


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
        n_epochs: int,
        early_stopping_params: EarlyStoppingParams,
        loss_fn: Callable,
        loss_grad_fn: Callable,
        loss_optx: Callable,
        optx_config: Dict,
        max_iter_optx: int,
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

    # Initialise early stopper and epoch counter
    early_stopper = None
    epoch = 0

    # Split model into parameters and static components - which might or might not include
    # the mechanistic parameters (i.e. dynamic_parameters)
    if conf.freeze_mechanistic_params:
        # Initialise filter_spec with True for all arrays
        filter_spec = jtu.tree_map(lambda x: eqx.is_array(x), model)
        # Freeze dynamic (mechanistic) parameters
        filter_spec = eqx.tree_at(
            lambda x: x.func.dynamic_parameters,
            filter_spec,
            replace=jax.tree_map(lambda _: False, filter_spec.func.dynamic_parameters)
        )
        # and partition accordingly
        params, static = eqx.partition(model, filter_spec)
    else:
        params, static = eqx.partition(model, eqx.is_array)
    # Get schedule
    schedule = get_scheduler(conf=conf, n_epochs=n_epochs)

    # Initialise optimiser and its state
    if not conf.use_lookahead:
        opt = optimisers[conf.optimiser](schedule)
        opt_state = opt.init(eqx.filter(model, eqx.is_array))
        make_step_fn = make_step
    else:
        opt = optax.lookahead(
            fast_optimizer=optimisers[conf.optimiser](schedule),
            sync_period=conf.lookahead_sync,
            slow_step_size=conf.lookahead_step_size,
            reset_state=conf.lookahead_reset_state,
        )
        # Initialise params for lookahead (now params has params.fast and params.slow)
        params = optax.LookaheadParams.init_synced(params)
        opt_state = opt.init(params)
        make_step_fn = make_step_looahead

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

    # Training loop
    for epoch in range(n_epochs + 1):
        params, opt_state, loss_train, grads = make_step_fn(
            params=params,
            static=static,
            opt=opt,
            opt_state=opt_state,
            input_data=training_data,
            u0=u0,
            t_save=t_save_train,
            t_span=t_span_train,
            conf=conf,
            loss_grad_fn=loss_grad_fn,
        )
        if conf.use_lookahead:
            model = eqx.combine(params.fast, static)
        else:
            model = eqx.combine(params, static)

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
                    **log_model_stats(model, grads, conf),
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
            save_plot=conf.save_plots,
            plot_dir=plot_dir,
            plot_fmt="svg"
        )

    wandb.log({"final_epoch": epoch})

    if max_iter_optx > 0:  # otherwise end training and return trained model
        # Define options and tags for BFGS solver
        options = {
            "rtol": optx_config["rtol"],
            "atol": optx_config["atol"],
        }
        tags = frozenset()
        # Additional training with BFGS via optimistix
        solver = optx.BFGS(**options)
        # Set dropout behaviour to be deterministic
        model_optx = model.copy(
                new_mlp=eqx.nn.inference_mode(model.func.mlp, value=True)
            )
        # Split model_optx in learnable params and static component
        if conf.freeze_mechanistic_params:
            params, static = eqx.partition(model_optx, filter_spec)
        else:
            params, static = eqx.partition(model_optx, eqx.is_array)
        # Minimise via Optimistix - no access to params during solve
        # sol = optx.minimise(
        #     fn=loss_optx,
        #     solver=solver,
        #     y0=params,
        #     args=(static, u0, t_save_train, t_span_train, training_data),
        #     max_steps=max_iter_optx,
        # )
        # Define additional arguments to pass optimistix loss function (train and val)
        args = (static, u0, t_save_train, t_span_train, training_data)
        args_val = (static, u0, t_save_val, t_span_val, validation_data)
        # Shape+dtype of the output of `fn`
        f_struct = jax.ShapeDtypeStruct((), jnp.float64)
        aux_struct = None

        sol = optx_solve(
            y=params, solver=solver,
            fn=loss_optx, args=args, args_val=args_val,
            f_struct=f_struct, aux_struct=aux_struct,
            options=options, tags=tags,
            max_steps=max_iter_optx,
            starting_epoch=epoch,
            conf=conf,
        )

        # Get final model
        model = eqx.combine(sol.value, static)

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
                save_plot=conf.save_plots,
                plot_dir=plot_dir,
                plot_fmt="svg",
            )

    wandb.finish()
    return model
