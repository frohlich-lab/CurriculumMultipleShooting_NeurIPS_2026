import jax.numpy as jnp
import numpy as np

from diffrax import SaveAt
from src.common import activation_functions, Conf, default_out_sizes, normalise, u0s
from src.model_examples.lotka_volterra import setup_diffrax_model as setup_lv
from src.model_examples.sir import setup_diffrax_model as setup_sir
from src.model_training.model_util import set_weight_bias
from src.model_training.losses import (compute_loss, compute_loss_grad, compute_loss_optx,
                                       compute_batch_loss, compute_batch_loss_grad, compute_batch_loss_optx)
from src.ude_problem import build_mlp


def get_io_size(conf, u0):
    if conf.dynamical_model == 'lotka_volterra':
        in_size = len(u0)
    elif conf.dynamical_model == 'sir':
        in_size = len(u0) - 1
    else:
        raise ValueError(f"Unknown dynamical model: {conf.dynamical_model}!")
    out_size = default_out_sizes[conf.dynamical_model] + conf.node_augmentation
    return in_size, out_size


def get_data_and_funcs_for_train(conf: Conf, dynamic_params_shape: tuple, u0) -> tuple:
    if conf.dynamical_model == 'lotka_volterra':
        # TODO @GiacomoFabrini find better way to express filepath and pass it as argument
        (train_data, t_save_train, t_span_train,
         validation_data, t_save_validation, t_span_validation) = setup_lv.get_training_validation_data()

        optx_config = {
            "rtol": 1e-6,  # changed from 1e-3
            "atol": 1e-6,  # changed from 1e-3
        }
    elif conf.dynamical_model == 'sir':
        # TODO @GiacomoFabrini find better way to express filepath and pass it as argument - same as above
        (train_data, t_save_train, t_span_train,
         validation_data, t_save_validation, t_span_validation) = setup_sir.get_training_validation_data()

        optx_config = {
            "rtol": 1e-6,  # changed from 1e-4
            "atol": 1e-6,  # changed from 1e-4
        }
    else:
        raise ValueError(f"Unknown dynamical model: {conf.dynamical_model}!")

    if conf.shooting_mode != "window_batch":
        loss_fn = compute_loss
        loss_grad_fn = compute_loss_grad
        loss_optx = compute_loss_optx
    else:
        loss_fn = compute_batch_loss
        loss_grad_fn = compute_batch_loss_grad
        loss_optx = compute_batch_loss_optx

    options = {
        "dynamic_params_scale": conf.dynamic_params_scale,
        "bounded_tanh_params": broadcast_bounds(
            bounds={"lb": conf.dynamic_params_lb, "ub": conf.dynamic_params_ub},
            shape=dynamic_params_shape
        )  # bounds are ignored if not using bounded tanh transform from Philipps et al. 2024
    }
    if conf.normalise_data:
        # Normalise initial condition u0 as well
        train_data, validation_data, u0 = normalise(train_data, validation_data, provide_u0=True, u0=u0)
    if conf.normalise_timepoints:
        # SaveAt
        t_save_train, t_save_validation = normalise(t_save_train.subs.ts, t_save_validation.subs.ts)
        t_save_train, t_save_validation = SaveAt(ts=t_save_train), SaveAt(ts=t_save_validation)
        # Span
        t_span_train = [t_save_train.subs.ts[0], t_save_train.subs.ts[-1]]
        t_span_validation = [t_save_train.subs.ts[0], t_save_validation.subs.ts[-1]]

    u0_train, u0_val = u0, u0
    # Store original format (possibly normalised) data and initial conditions for plotting
    plot_args = {
        "u0": u0,
        "t_save_train": t_save_train,
        "t_span_train": t_span_train,
        "training_data": train_data,
        "t_save_val": t_save_validation,
        "t_span_val": t_span_validation,
        "validation_data": validation_data,
    }
    # Handle batched windowing
    if conf.shooting_mode == "window_batch":
        # Training data
        train_data, t_save_train, t_span_train = window_batch_dataloader(
            input_data=train_data,
            t_save=t_save_train,
            window_length=conf.shooting_window_length,
            batch_size=conf.batch_size,  # default: 64
            random_seed=conf.multistart_job,
        )
        # Get initial condition batches
        u0_train = jnp.array([data_batch[0] for data_batch in train_data])
        # Batch validation data as well (fictitious expanded dimension)
        u0_val = jnp.expand_dims(u0_val, axis=0)
        validation_data = jnp.expand_dims(validation_data, axis=0)
        if isinstance(t_save_validation, SaveAt):
            t_save_validation = t_save_validation.subs.ts
        t_save_validation = jnp.expand_dims(jnp.array(t_save_validation), axis=0)
        t_span_validation = jnp.expand_dims(jnp.array(t_span_validation), axis=0)

    # Pack arguments to return
    train_args = train_data, t_save_train, t_span_train, u0_train
    val_args = validation_data, t_save_validation, t_span_validation, u0_val
    loss_args = loss_fn, loss_grad_fn, loss_optx
    option_args = optx_config, options
    return train_args, val_args, loss_args, option_args, plot_args


def broadcast_bounds(bounds, shape):
    return {
        key: jnp.broadcast_to(value, shape)
        for key, value in bounds.items()
    }


def get_initial_conditions(conf: Conf):
    u0 = u0s[conf.dynamical_model]
    # Apply NODE-like augmentation
    if conf.node_augmentation > 0:
            u0 = jnp.concatenate([u0, jnp.zeros(conf.node_augmentation)])
    # Ensure 64-bit precision
    u0 = u0.astype(jnp.float64)
    return u0


def setup_mlp(conf: Conf, u0: jnp.ndarray, key_mlp, key_init):
    # Define input and output sizes for MLP
    in_size, out_size = get_io_size(conf, u0)

    # Check activation function is valid
    if conf.activation_fn_name not in activation_functions:
        raise ValueError(f"Unknown activation function: {conf.activation_fn_name}!")

    # Build the MLP, initialise its weights and biases
    mlp = set_weight_bias(
        build_mlp(
            in_size=in_size,
            out_size=out_size,
            width_size=conf.mlp_layer_width,
            hidden_layers=conf.mlp_depth,
            key=key_mlp,
            use_bias=conf.use_bias,
            use_final_bias=conf.use_final_bias,
            activation=activation_functions[conf.activation_fn_name],
            final_activation=activation_functions[conf.final_activation_fn_name],
            use_dropout=conf.use_dropout,
            input_dropout_rate=conf.input_dropout_rate,
            hidden_dropout_rate=conf.hidden_dropout_rate,
            use_skip_connections=conf.use_skip_connections,
        ),
        conf.init_scale,
        key_init
    )
    return mlp


def window_batch_dataloader(input_data, t_save, t_span, window_length: int, batch_size: int, random_seed: int):
    """
    Returns random batches of sequences from input_data along with t_save and t_span batches.

    Parameters:
    input_data (jnp.array):
        Array of training data.
    t_save (jnp.array):
        Array of evaluation timepoints.
    t_span (list):
        List with first and last timepoints [t_start, t_end].
    window_length (int):
        Number of consecutive timepoints in each sequence.
    batch_size (int):
        Number of sequences in each batch.

    Returns:
    windowed_batches (jnp.array):
        Batch of input data sequences of shape (batch_size, window_length).
    t_save_batches (jnp.array):
        Corresponding evaluation timepoints for each window.
    t_span_batches (jnp.array):
        Start and end time for each window.
    """
    # Ensure the data is large enough for the requested window and batch size
    data_length = len(input_data)
    if window_length > data_length:
        raise ValueError("window_length cannot be larger than the length of the input_data")
    max_start_idx = data_length - window_length

    # Set random seed
    np.random.seed(random_seed)
    # Randomly sample starting indices for the windows (with replacement)
    start_indices = np.random.choice(max_start_idx, size=batch_size, replace=True)
    start_indices = np.array(list(set(start_indices.tolist())))  # remove duplicates

    # Extract windows for input_data and t_save
    windowed_batches = jnp.array([
        input_data[start : start + window_length] for start in start_indices
    ])
    if isinstance(t_save, SaveAt):
        t_save = t_save.subs.ts
    # Avoid using SaveAt types to use jax.vmap across batches
    t_save_batches = jnp.array([
        t_save[start : start + window_length] for start in start_indices
    ])

    # Calculate t_span for each window
    t_span_batches = jnp.array([
        [t_save_batch[0], t_save_batch[-1]] for t_save_batch in t_save_batches
    ])

    return windowed_batches, t_save_batches, t_span_batches
