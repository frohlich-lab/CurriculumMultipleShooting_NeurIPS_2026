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
from src.model_training.model_util import regularise_mlp, set_weight_bias
from src.model_training.training_util import (eig_sv_plot, generate_incremental_arrays, generate_log_epochs,
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


def train_optimistix(
        model: UDEProblem,
        filter_spec: PyTree,
        conf: Conf,
        optx_config: dict[str: float],
        loss_optx: Callable,
        epoch: int,
        num_epochs: int,
        u0: object,
        t_save_train: SaveAt, t_span_train: List, training_data: jnp.ndarray,
        t_save_val: SaveAt, t_span_val: List, validation_data: jnp.ndarray,
        frozen_mechanistic: bool = False, frozen_nn: bool = False,
) -> tuple[UDEProblem, SimpleNamespace, UDEProblem]:
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
    params, static = eqx.partition(model_optx, filter_spec)

    # Minimise via Optimistix - no access to params during solve
    # sol = optx.minimise(
    #     fn=loss_optx,
    #     solver=solver,
    #     y0=params,
    #     args=(static, u0, t_save_train, t_span_train, training_data),
    #     max_steps=max_iter_optx_finetune,
    # )

    # Define additional arguments to pass optimistix loss function (train and val) for step by step solve
    args = (static, u0, t_save_train, t_span_train, training_data, conf.reg_exp, conf.reg_strength)
    args_val = (static, u0, t_save_val, t_span_val, validation_data, conf.reg_exp, conf.reg_strength)
    # Shape+dtype of the output of `fn`
    f_struct = jax.ShapeDtypeStruct((), jnp.float64)
    aux_struct = None

    # Optimistix solve
    sol = optx_solve(
        y=params, solver=solver,
        fn=loss_optx, args=args, args_val=args_val,
        f_struct=f_struct, aux_struct=aux_struct,
        options=options, tags=tags,
        max_steps=num_epochs,
        starting_epoch=epoch,
        conf=conf,
        frozen_mechanistic=frozen_mechanistic,
        frozen_nn=frozen_nn,
    )

    # Get final model
    model = eqx.combine(sol.value, static)
    return model, sol, static


def optx_solve(
        y, solver, fn,
        args, args_val,
        f_struct, aux_struct,
        options, tags, max_steps,
        starting_epoch, conf,
        frozen_mechanistic, frozen_nn,
):
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
                        conf,
                        frozen_mechanistic, frozen_nn,
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
        t_save_val: SaveAt,
        t_span_val: List,
        conf: Conf,
        early_stopping_params: EarlyStoppingParams,
        loss_fn: Callable,
        loss_grad_fn: Callable,
        loss_optx: Callable,
        optx_config: Dict,
        max_iter_optx_mechanistic: int,
        n_epochs: int,
        max_iter_optx_finetune: int,
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
    :param max_iter_optx_mechanistic:
        Maximum number of iterations for initial fitting of mechanistic parameters (dynamical parameters).
    :param n_epochs:
        Number of training iterations (epochs).
    :param max_iter_optx_finetune:
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
    best_model, loss_val_min = model, jnp.inf

    # Create filter_spec allowing for training of mechanistic params only or neural network only
    filter_spec_freeze_mechanistic = jtu.tree_map(lambda x: eqx.is_array(x), model)
    # Freeze dynamic (mechanistic) parameters
    filter_spec_freeze_mechanistic = eqx.tree_at(
        lambda x: x.func.dynamic_parameters,
        filter_spec_freeze_mechanistic,
        replace=jax.tree_map(lambda _: False, filter_spec_freeze_mechanistic.func.dynamic_parameters)
    )
    # Freeze neural network
    filter_spec_freeze_nn = jtu.tree_map(lambda x: False, model)
    filter_spec_freeze_nn = eqx.tree_at(
        lambda x: x.func.dynamic_parameters,
        filter_spec_freeze_nn,
        replace=jax.tree_map(lambda _: True, filter_spec_freeze_nn.func.dynamic_parameters)
    )


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

    # Extract initialised MLP from model
    initialised_mlp = model.func.mlp

    # STAGE 1: fit mechanistic model parameters -- do not use curriculum learning
    if max_iter_optx_mechanistic > 0:
        # Zero-out the MLP contribution to train mechanistic parameters
        zeroed_mlp = set_weight_bias(initialised_mlp, 0.0, jax.random.PRNGKey(0))
        # In case dropout is enabled, set inference to True
        model = model.copy(
            new_mlp=eqx.nn.inference_mode(zeroed_mlp, value=True)
        )
        # Freeze neural network
        model, sol, static = train_optimistix(
            model=model,
            filter_spec=filter_spec_freeze_nn,
            conf=conf,
            optx_config=optx_config,
            loss_optx=loss_optx,
            epoch=epoch,
            num_epochs=max_iter_optx_mechanistic,
            u0=u0,
            t_save_train=t_save_train,
            t_span_train=t_span_train,
            training_data=training_data,
            t_save_val=t_save_val,
            t_span_val=t_span_val,
            validation_data=validation_data,
            frozen_mechanistic=False,
            frozen_nn=True,
        )

        # Update epoch count
        epoch = sol.stats["num_steps"]

        # Compute validation loss of optimistix trained model
        loss_val = loss_fn(
            model,
            u0,
            t_save_val,
            t_span_val,
            validation_data,
            conf.reg_exp,
            conf.reg_strength,
        )

        # Update best_model
        if loss_val < loss_val_min:
            best_model = model

        # Plot eigenvalue and singular value spectrum
        # eig_sv_plot(
        #     sol.value,
        #     static,
        #     u0,
        #     t_save_train, t_span_train, training_data,
        #     conf,
        #     epoch,
        # )

        # Log plots again after further training with optimistix/BFGS
        for dataset, t_save, t_span, data in zip(
                ["train", "validation"],
                [t_save_train, t_save_val],
                [t_span_train, t_span_val],
                [training_data, validation_data]
        ):
            df_predict, df_obs = get_fit_dfs(model, u0, t_save, t_span, data)
            plot_dir = plots_dir / conf.dynamical_model / "mechanistic_optx"
            plot_dir.mkdir(exist_ok=True, parents=True)
            log_plots(
                df_predict,
                df_obs,
                labels,
                colors,
                epoch,
                f"{dataset}_mechanistic_optx",
                save_plot=conf.save_plots,
                plot_dir=plot_dir,
                plot_fmt="svg",
            )
        # Increment epoch before moving to next stage
        epoch +=1


    # STAGE 2: Train with ADAM/ADAM-W -- freeze mechanistic model parameters
    if n_epochs > 0:
        # Restore the previously initialised MLP
        model = model.copy(
            new_mlp=initialised_mlp
        )
        # Freeze mechanistic parameters
        params, static = eqx.partition(model, filter_spec_freeze_mechanistic)

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

        # Generate training data - curriculum learning or not
        incremental_training_data, incremental_train_save, incremental_t_span = generate_incremental_arrays(
            conf,
            training_data,
            t_save_train,
            n_epochs, 10,
        )

        # Generate regularly log-spaced epochs for early-stopping evaluation + model stat logging - from DMMs
        log_epochs = generate_log_epochs(n_epochs=n_epochs, num_samples=100, min_dist=5)

        # Training loop
        for nn_epoch in range(n_epochs):
            params, opt_state, loss_train, grads = make_step(
                params=params,
                static=static,
                opt=opt,
                opt_state=opt_state,
                input_data=incremental_training_data[nn_epoch],
                u0=u0,
                t_save=incremental_train_save[nn_epoch],
                t_span=incremental_t_span[nn_epoch],
                conf=conf,
                loss_grad_fn=loss_grad_fn,
            )
            # Recombine dynamic and static into full model
            model = eqx.combine(
                params.fast if conf.use_lookahead else params,
                static,
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

            # Keep track of best performing model on validation dataset (and corresponding loss)
            if loss_val < loss_val_min:
                best_model = inference_model
                loss_val_min = loss_val

            # log loss_train, loss_val, learning rate, param values and grads on log-spaced epochs
            if nn_epoch in log_epochs:
                wandb.log(
                    {
                        "loss_train": loss_train,
                        "loss_val": loss_val,
                        "learning_rate": schedule(nn_epoch),
                        "num_train_samples": len(incremental_training_data[nn_epoch]),
                        **log_model_stats(model, grads, conf, frozen_mechanistic=True, frozen_nn=False),
                        "max_grad_norm": max_grad_norm(grads),
                    },
                    step=epoch+nn_epoch
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
                        step=epoch+nn_epoch
                    )

                print(
                    f"epoch {nn_epoch} | "
                    f"loss_train {loss_train} | "
                    f"loss_val {loss_val} | "
                )

                # # TODO @GiacomoFabrini: why is this so slow on SIR?
                # if nn_epoch in log_epochs[::10]:  # every 10 log spaced epochs = 10 samples overall
                #     eigenvalues, singular_values = eig_sv_plot(
                #         params,
                #         static,
                #         u0, t_save_train, t_span_train,
                #         training_data,
                #         conf,
                #         epoch,
                #         plot=True,
                #         return_values=True,
                #     )
                #     wandb.log(
                #         {
                #             "max_eigenvalue": jnp.max(eigenvalues),
                #             "min_eigenvalue": jnp.min(eigenvalues),
                #             "max_sv": jnp.max(singular_values),
                #             "min_sv": jnp.min(singular_values),
                #             "condition_number": jnp.max(singular_values)/jnp.min(singular_values),
                #         },
                #         step=epoch+nn_epoch
                #     )

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
                        step=epoch+nn_epoch,
                    )
                    # Stop training if we have run out of patience
                    if early_stopper.should_stop:
                        print(f'Met early stopping criteria, breaking at epoch {epoch+nn_epoch}')
                        break

        # Plot fit (train/val) at end of training -- using best model on validation dataset
        for dataset, t_save, t_span, data in zip(
                ["train", "validation"],
                [t_save_train, t_save_val],
                [t_span_train, t_span_val],
                [training_data, validation_data]
        ):
            df_predict, df_obs = get_fit_dfs(best_model, u0, t_save, t_span, data)
            plot_dir = plots_dir / conf.dynamical_model / "optax"
            plot_dir.mkdir(exist_ok=True, parents=True)
            log_plots(
                df_predict,
                df_obs,
                labels,
                colors,
                epoch+nn_epoch,
                f"{dataset}",
                save_plot=conf.save_plots,
                plot_dir=plot_dir,
                plot_fmt="svg"
            )

        wandb.log({"final_epoch": epoch+nn_epoch})
        # Update epoch
        epoch += nn_epoch + 1

    # Train with BFGS via optimistix
    if max_iter_optx_finetune > 0:  # otherwise end training and return trained model
        model, sol, static = train_optimistix(
            model=model,
            filter_spec=eqx.is_array,
            conf=conf,
            optx_config=optx_config,
            loss_optx=loss_optx,
            epoch=epoch,
            num_epochs=max_iter_optx_finetune,
            u0=u0,
            t_save_train=t_save_train,
            t_span_train=t_span_train,
            training_data=training_data,
            t_save_val=t_save_val,
            t_span_val=t_span_val,
            validation_data=validation_data,
            frozen_mechanistic=False,
            frozen_nn=False,
        )

        # Compute validation loss of optimistix trained model
        loss_val = loss_fn(
            model,
            u0,
            t_save_val,
            t_span_val,
            validation_data,
            conf.reg_exp,
            conf.reg_strength,
        )

        # Update best_model
        if loss_val < loss_val_min:
            best_model = model

        # Plot eigenvalue and singular value spectrum
        # eig_sv_plot(
        #     sol.value,
        #     static,
        #     u0, t_save_train, t_span_train,
        #     training_data,
        #     conf,
        #     epoch+sol.stats["num_steps"],
        # )

        # Log plots again after further training with optimistix/BFGS
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
    # Return best_model on validation dataset
    return best_model
