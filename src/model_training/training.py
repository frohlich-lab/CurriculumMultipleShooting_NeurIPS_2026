import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import optimistix as optx
import subprocess
import wandb
import warnings

from src.common import Conf, EarlyStoppingParams, plots_dir
from diffrax import DirectAdjoint, SaveAt
from flax.training.early_stopping import EarlyStopping
from jaxtyping import Array, Float, PyTree
from src.model_examples.lotka_volterra.setup_diffrax_model import (get_fit_dfs as get_fit_dfs_lv,
                                                                   compute_hess_array as compute_hess_array_lv)
from src.model_examples.sir.setup_diffrax_model import (get_fit_dfs as get_fit_dfs_sir,
                                                        compute_hess_array as compute_hess_array_sir)
from src.model_training.model_util import regularise_mlp
from src.model_training.training_util import (eig_sv_plot, generate_incremental_arrays, get_model_after_step,
                                              get_opt_opt_state_params, get_scheduler,
                                              initialise_wandb, log_model_stats, log_plots,
                                              make_grads_finite, max_grad_norm)
from src.ude_problem import UDEProblem
from types import SimpleNamespace
from typing import Callable, Dict, List, Union

# Some FutureWarning with unhashable DynamicJaxprTracer object in the diffeq solve/call
warnings.simplefilter(action='ignore', category=FutureWarning)


def apply_hessian_preconditioning_to_grads(
        grads: Union[PyTree, UDEProblem],
        params: Union[PyTree, UDEProblem],
        static: Union[PyTree, UDEProblem],
        args: tuple,
        conf: Conf,
):
    # define hessian functions for different dynamical models
    hessian_fn = {
        "lotka_volterra": compute_hess_array_lv,
        "sir": compute_hess_array_sir,
    }
    # unpack args
    u0, t_save, t_span, input_data = args
    # flatten params PyTree and get unflatten function
    flat_p, unflatten_p = jax.flatten_util.ravel_pytree(params)
    # compute hessian as a 2D array
    hessian = hessian_fn[conf.dynamical_model](
        flat_p, unflatten_p, static,
        u0, t_save, t_span, input_data,
        DirectAdjoint(),
        conf.reg_exp, conf.reg_strength,
    )

    # Keep only the diagonal (experiment - Sebastian) -- disabled
    # hessian_diag = hessian * jnp.eye(hessian.shape[0])

    # Compute the inverse of the Hessian
    inv_hessian = jnp.linalg.pinv(hessian)

    # Flatten the grads PyTree
    flat_grads, unravel_grads = jax.flatten_util.ravel_pytree(grads)
    # Multiply the flat gradients by the inverse Hessian to precondition gradients
    flat_precond_grads = jnp.dot(inv_hessian, flat_grads)
    # Unflatten the preconditioned gradients back to the PyTree structure + ensure they are finite
    grads = make_grads_finite(unravel_grads(flat_precond_grads))
    return grads


def optx_solve(y, solver, fn, args, args_val, f_struct, aux_struct, options, tags, max_steps, starting_epoch, conf):
    # Initialise loss variables
    loss_train, loss_val = jnp.inf, jnp.inf
    aux = None

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
        params: Union[PyTree, UDEProblem],
        static: Union[PyTree, UDEProblem],
        opt: optax.GradientTransformation,
        opt_state: PyTree,
        input_data: Float[Array, '...'],  # TODO @GiacomoFabrini fix input data shape
        u0: jnp.ndarray,
        t_save: List,
        t_span: List,
        conf: Conf,
        loss_grad_fn: Callable,
):
    if conf.use_lookahead:
        opt_params = params.fast
    else:
        opt_params = params
    loss_value, grads = loss_grad_fn(
        opt_params,
        static,
        u0,
        t_save,
        t_span,
        input_data,
        conf.reg_exp,
        conf.reg_strength,
    )
    grads = make_grads_finite(grads)
    if conf.hessian_precondition:
        args = (u0, t_save, t_span, input_data)
        grads = apply_hessian_preconditioning_to_grads(grads, opt_params, static, args, conf)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = eqx.apply_updates(params, updates)
    return params, opt_state, loss_value, grads


def train(
        model: UDEProblem,
        u0: jnp.ndarray,
        training_data: jnp.ndarray,
        t_save_train: SaveAt,
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

    # Initialise variables
    early_stopper, opt, opt_state, schedule = None, None, None, None
    epoch = 0

    # Split model into params and static - depends on whether we are freezing mechanistic parameters or learning them
    if conf.freeze_mechanistic_params:
        # Initialise filter_spec with True for all arrays
        filter_spec = jtu.tree_map(lambda x: eqx.is_array(x), model)
        # Freeze dynamic (mechanistic) parameters
        filter_spec = eqx.tree_at(
            lambda x: x.func.dynamic_parameters,
            filter_spec,
            replace=jax.tree_map(lambda _: False, filter_spec.func.dynamic_parameters)
        )
    else:
        filter_spec = eqx.is_array
    params, static = eqx.partition(model, filter_spec)

    # Define plotting functions, labels and colours
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

    # Train with ADAM/ADAM-W
    if n_epochs > 0:
        # Get schedule
        schedule = get_scheduler(conf=conf, n_epochs=n_epochs)

        # Initialise optimiser, optimiser state + update params if using lookahead
        opt, opt_state, params = get_opt_opt_state_params(conf, params, schedule)

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

        incremental_training_data, incremental_train_save, incremental_t_span = generate_incremental_arrays(
            conf,
            training_data,
            t_save_train,
            n_epochs, 10,
        )
        # Training loop
        for epoch in range(n_epochs):
            params, opt_state, loss_train, grads = make_step(
                params=params,
                static=static,
                opt=opt,
                opt_state=opt_state,
                input_data=incremental_training_data[epoch],
                u0=u0,
                t_save=incremental_train_save[epoch],
                t_span=incremental_t_span[epoch],
                conf=conf,
                loss_grad_fn=loss_grad_fn,
            )
            model = get_model_after_step(params, static, conf)

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

            # Plot eigenvalue and singular value spectrum at first and last ADAM epoch
            if epoch in [0, n_epochs-1]:
                eig_sv_plot(
                    params,
                    static,
                    u0, t_save_train, t_span_train,
                    training_data,
                    conf,
                    epoch,
                    plot=True,
                )

            # log loss_train, loss_val, learning rate, param values and grads every 5 epochs
            if epoch % 10 == 0:
                wandb.log(
                    {
                        "loss_train": loss_train,
                        "loss_val": loss_val,
                        "learning_rate": schedule(epoch),
                        "num_train_samples": len(incremental_training_data[epoch]),
                        **log_model_stats(model, grads, conf),
                        "max_grad_norm": max_grad_norm(grads),
                    },
                    step=epoch
                )

                # Log regularisation term if applied
                if conf.reg_strength > 0:
                    wandb.log(
                        {
                            "regularisation_loss": regularise_mlp(
                                model.func.mlp,
                                conf.reg_exp,
                                conf.reg_strength
                            )
                        },
                        step=epoch
                    )

                print(
                    f"epoch {epoch} | "
                    f"loss_train {loss_train} | "
                    f"loss_val {loss_val} | "
                )

                if epoch % 50 == 0:  # only compute every 50 epochs (1 every 5 eval epochs)
                    eigenvalues, singular_values = eig_sv_plot(
                        params,
                        static,
                        u0, t_save_train, t_span_train,
                        training_data,
                        conf,
                        epoch,
                        plot=False,
                    )
                    wandb.log(
                        {
                            "max_eigenvalue": jnp.max(eigenvalues),
                            "min_eigenvalue": jnp.min(eigenvalues),
                            "max_sv": jnp.max(singular_values),
                            "min_sv": jnp.min(singular_values),
                            "condition_number": jnp.max(singular_values)/jnp.min(singular_values),
                        },
                        step=epoch
                    )

                # Check early stopping criterion
                if early_stopping_params.use_early_stopping:
                    # Update early stopper
                    early_stopper = early_stopper.update(loss_val)
                    # Debugging statements
                    print(
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

    # Train with BFGS via optimistix
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
        args = (static, u0, t_save_train, t_span_train, training_data, conf.reg_exp, conf.reg_strength)
        args_val = (static, u0, t_save_val, t_span_val, validation_data, conf.reg_exp, conf.reg_strength)
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

        # Plot eigenvalue and singular value spectrum
        eig_sv_plot(
            sol.value,
            static,
            u0, t_save_train, t_span_train,
            training_data,
            conf,
            epoch+sol.stats["num_steps"],
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
                save_plot=conf.save_plots,
                plot_dir=plot_dir,
                plot_fmt="svg",
            )

    # Get W&B run directory
    wandb_stripped_dir = wandb.run.dir.rsplit('/files', 1)[0]
    command = f"wandb sync {wandb_stripped_dir}"
    wandb.finish()
    # Log everything to W&B after end of run
    try:
        _ = subprocess.run(command, shell=True)
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Error syncing wandb directory: {e}")
    return model
