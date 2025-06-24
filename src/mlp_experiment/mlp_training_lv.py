import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import wandb

from dataclasses import replace
from jaxtyping import Array, Float, PyTree
from src.common import Conf
from src.model_training.model_util import regularise_mlp, get_mse
from src.model_training.training import make_grads_finite
from src.model_training.training_util import (get_hessian_metrics, generate_log_epochs,
                                              get_opt_opt_state_params, get_scheduler, initialise_wandb)
from typing import Union


@eqx.filter_jit
def compute_mlp_loss(params, static, data, targets, reg_exp, reg_strength):
    model = eqx.combine(params, static)
    y_pred = jax.vmap(model)(data)
    return get_mse(pred=y_pred, data=targets) + regularise_mlp(mlp=model, reg_exp=reg_exp, reg_strength=reg_strength)


def compute_mlp_loss_grad(params, static, data, targets, reg_exp, reg_strength):
    return eqx.filter_value_and_grad(compute_mlp_loss, has_aux=False)(
        params, static, data, targets, reg_exp, reg_strength
    )


@eqx.filter_jit
@jax.hessian
def compute_mlp_hessian(
        flat_p, unflatten_p, static, data, targets, mlp_reg_exp, mlp_reg_strength,
):
    return compute_mlp_loss(
        unflatten_p(flat_p), static, data, targets, mlp_reg_exp, mlp_reg_strength
    )


@eqx.filter_jit
def mlp_training_step(
        params: Union[PyTree, eqx.Module],
        static: Union[PyTree, eqx.Module],
        opt: optax.GradientTransformation,
        opt_state: PyTree,
        input_data: Float[Array, '...'],  # TODO @GiacomoFabrini fix input and output data shape
        targets: Float[Array, '...'],
        conf: Conf,
):
    loss_value, grads = compute_mlp_loss_grad(
        params,
        static,
        input_data,
        targets,
        conf.mlp_reg_exp,
        conf.mlp_reg_strength,
    )
    grads = make_grads_finite(grads)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = eqx.apply_updates(params, updates)
    return params, opt_state, loss_value, grads


def mlp_train(
        model: eqx.nn.MLP,
        training_data: jnp.ndarray,
        training_targets: jnp.ndarray,
        validation_data: jnp.ndarray,
        validation_targets: jnp.ndarray,
        conf: Conf,
        n_epochs: int,
        use_wandb: bool = True,
):
    # Set training setting in configuration
    conf = replace(conf, training_setting="MLP")
    # Initialise wandb and define metrics
    if use_wandb:
        initialise_wandb(
            conf=conf,
            model=model,
        )

    # Initialise epoch counter
    epoch = 0

    # Get schedule and partition model
    schedule = get_scheduler(conf=conf, n_epochs=n_epochs)
    params, static = eqx.partition(model, eqx.is_array)

    # Initialise optimiser and optimiser state
    opt, opt_state, params = get_opt_opt_state_params(conf, params, schedule)

    # Generate log-spaced epochs
    log_epochs = generate_log_epochs(n_epochs=n_epochs, num_samples=100, min_dist=5)
    # Add first and final epochs
    log_epochs = jnp.unique(
        jnp.sort(
            jnp.concatenate(
                [log_epochs, jnp.array([0, n_epochs - 1])]
            )
        )
    )

    # Training loop
    for epoch in range(n_epochs + 1):
        params, opt_state, loss_train, grads = mlp_training_step(
            params=params,
            static=static,
            opt=opt,
            opt_state=opt_state,
            input_data=training_data,
            targets=training_targets,
            conf=conf,
        )
        model = eqx.combine(params, static)

        if conf.use_dropout:
            # Get a copy of the model for inference
            inference_model = model.replace_mlp(
                new_mlp=model.func.mlp,
                switch_to_inference=True,
            )
        else:
            # no need for difference in training/inference behaviour
            inference_model = model

        inference_params, inference_static = eqx.partition(inference_model, eqx.is_array)

        # compute validation loss
        loss_val = compute_mlp_loss(
            inference_params,
            inference_static,
            validation_data,
            validation_targets,
            conf.mlp_reg_exp,
            conf.mlp_reg_strength,
        )

        # log loss_train, loss_val, learning rate every 5 epochs
        if epoch in log_epochs:
            if use_wandb:
                wandb.log(
                    {
                        "loss_train": loss_train,
                        "loss_val": loss_val,
                        "learning_rate": schedule(epoch),
                    },
                    step=epoch
                )
            if conf.compute_hessian_metrics:
                # flatten params PyTree and get unflatten function
                flat_p, unflatten_p = jax.flatten_util.ravel_pytree(params)
                # Compute Hessian
                hessian = compute_mlp_hessian(
                    flat_p, unflatten_p, static,
                    training_data, training_targets,
                    conf.mlp_reg_exp, conf.mlp_reg_strength,
                )
                if use_wandb:
                    # Log Hessian
                    wandb.log(
                        {
                            **get_hessian_metrics(hessian, return_eigs_svs=False),
                        },
                        step=epoch
                    )

    if use_wandb:
        wandb.log({"final_epoch": epoch})

    # Rudimentary plot - log to wandb
    plt.subplots(1, 3, figsize=(9, 4), sharey=True, sharex=True)
    for data, targets, label, i in zip(
            [training_data, validation_data],
            [training_targets, validation_targets],
            ['Training', 'Validation'],
            [1, 2]
    ):
        plt.subplot(1, 3, i)
        plt.plot(targets, 's', label=f'{label} targets')
        plt.plot(jax.vmap(model)(data), '-', label='Model fit')
        plt.title(label)
        plt.legend()
    plt.subplot(133)
    plt.plot(training_targets[:, 0], jax.vmap(model)(training_data)[:, 0], 'o')
    plt.plot(training_targets[:, 1], jax.vmap(model)(training_data)[:, 1], 's')
    plt.title('Prediction (y) vs target (x)')
    if use_wandb:
        wandb.log({f"mlp_results": plt}, step=epoch+1)
        wandb.finish()

    return model
