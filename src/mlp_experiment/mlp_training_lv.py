import equinox as eqx
import git
import jax
import jax.numpy as jnp
import itertools as itt
import matplotlib.pyplot as plt
import optax
import wandb

from flax.training.early_stopping import EarlyStopping
from jaxtyping import Array, Float, PyTree
from src.common import Conf, EarlyStoppingParams
from src.model_training.model_util import regularise_mlp
from src.model_training.training import optimisers, make_grads_finite
from src.model_training.training_util import get_scheduler
from typing import Callable


@eqx.filter_jit
def compute_loss(model, data, targets, reg_exp, reg_strength):
    y_pred = jax.vmap(model)(data)
    return jnp.sum((y_pred - targets) ** 2) + regularise_mlp(model, reg_exp, reg_strength)


@eqx.filter_value_and_grad
def compute_loss_grad(params, static, data, targets, reg_exp, reg_strength):
    model = eqx.combine(params, static)
    y_pred = jax.vmap(model)(data)
    return jnp.sum((y_pred - targets) ** 2) + regularise_mlp(model, reg_exp, reg_strength)


@eqx.filter_jit
def mlp_training_step(
        params: eqx.Module,
        static: eqx.Module,
        opt: optax.GradientTransformation,
        opt_state: PyTree,
        input: Float[Array, '...'],  # TODO @GiacomoFabrini fix input and output data shape
        targets: Float[Array, '...'],
        conf: Conf,
        loss_grad_fn: Callable,
):
    loss_value, grads = loss_grad_fn(
        params,
        static,
        input,
        targets,
        conf.reg_exp,
        conf.reg_strength,
    )
    grads = make_grads_finite(grads)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = eqx.apply_updates(params, updates)
    return params, opt_state, loss_value, grads


@eqx.filter_jit
def mlp_training_step_lookahead(
        params: eqx.Module,
        static: eqx.Module,
        opt: optax.GradientTransformation,
        opt_state: PyTree,
        input_data: Float[Array, '...'],  # TODO @GiacomoFabrini fix input data shape
        conf: Conf,
        loss_grad_fn: Callable,
):
    loss_value, grads = loss_grad_fn(
        params.fast,
        static,
        input_data,
        conf.reg_exp,
        conf.reg_strength,
    )
    grads = make_grads_finite(grads)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = eqx.apply_updates(params, updates)
    return params, opt_state, loss_value, grads


def initialise_wandb_mlp(
        conf: Conf,
        early_stopping_params: EarlyStoppingParams,
        model: eqx.nn.MLP,
):
    # Link to repo
    repo = git.Repo(search_parent_directories=True)
    # Initialise run
    wandb.init(
        project=f"TestUDE_LV_MLP",
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
    }
    for metric in metrics.keys():
        # if metric summary not specified
        if metrics[metric] is None:
            wandb.define_metric(metric)
        else:
            wandb.define_metric(metric, summary=metrics[metric])
    # Define layer statistics (weight/bias value and grads)
    # from Fabian's NeuralCoarseGraining
    for il, _ in enumerate(model.layers):
        for par_label, val_label in itt.product(('w', 'b'), ('vals', 'grads')):
            wandb.define_metric(
                f'layer{il}.{par_label}_{val_label}',
                summary='last',
            )


def mlp_train(
    model: eqx.nn.MLP,
    training_data: jnp.ndarray,
    training_targets: jnp.ndarray,
    validation_data: jnp.ndarray,
    validation_targets: jnp.ndarray,
    conf: Conf,
    n_epochs: int,
    early_stopping_params: EarlyStoppingParams,
    loss_fn: Callable,
    loss_grad_fn: Callable,
):
    # Initialise wandb and define metrics
    initialise_wandb_mlp(
        conf=conf,
        early_stopping_params=early_stopping_params,
        model=model,
    )

    # Initialise early stopper and epoch counter
    early_stopper = None
    epoch = 0

    # Get schedule and partition model
    schedule = get_scheduler(conf=conf, n_epochs=n_epochs)
    params, static = eqx.partition(model, eqx.is_array)

    # Initialise optimiser and its state
    if not conf.use_lookahead:
        opt = optimisers[conf.optimiser](schedule)
        opt_state = opt.init(eqx.filter(model, eqx.is_array))
        make_step_fn = mlp_training_step
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
        make_step_fn = mlp_training_step_lookahead

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
            input=training_data,
            targets=training_targets,
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
            validation_data,
            validation_targets,
            conf.reg_exp,
            conf.reg_strength,
        )

        # log loss_train, loss_val, learning rate every 5 epochs
        if epoch % 5 == 0:
            wandb.log(
                {
                    "loss_train": loss_train,
                    "loss_val": loss_val,
                    "learning_rate": schedule(epoch),
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

    wandb.log({"final_epoch": epoch})
    wandb.finish()
    # Rudimentary plot
    plt.subplots(1,2, figsize=(10,5))
    plt.subplot(131)
    plt.plot(training_data, 'o', label='Training data')
    plt.plot(training_targets, 's', label='Targets')
    plt.plot(jax.vmap(model)(training_data), '-', label='Model fit')
    plt.title('Training')
    plt.legend()
    plt.subplot(132)
    plt.plot(validation_data, 'o', label='Training data')
    plt.plot(validation_targets, 's', label='Targets')
    plt.plot(jax.vmap(model)(validation_data), '-', label='Model fit')
    plt.title('Validation')
    plt.legend()
    plt.subplot(133)
    plt.plot(training_targets[:, 0], jax.vmap(model)(training_data)[:, 0], 'o')
    plt.plot(training_targets[:, 1], jax.vmap(model)(training_data)[:, 1], 's')
    plt.title('Prediction (y) vs target (x)')
    plt.show()
    return model
