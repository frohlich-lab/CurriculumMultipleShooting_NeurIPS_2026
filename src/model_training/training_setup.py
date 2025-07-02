import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diffrax import SaveAt
from functools import partial
from jaxtyping import Array
from pathlib import Path
from src.common import (
    activation_functions,
    Conf,
    default_out_sizes,
    normalise,
    plots_dir,
    state_dims,
    TrainValArgs,
    true_dynamic_parameters,
    u0s
)
from src.model_examples.lotka_volterra import setup_lotka_volterra_model as setup_lv
from src.model_examples.sir import setup_sir_model as setup_sir
from src.model_training.model_util import bounded_tanh_transform_params, set_weight_bias
from src.model_training.losses import (
    compute_validation_loss, compute_loss, compute_loss_grad,
    compute_val_loss_optx, compute_loss_optx,
)
from src.ude_problem import build_mlp
from typing import Optional, Tuple, Union


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
        base_loss_fn = partial(setup_lv.lv_base_loss, state_dimension=state_dims[conf.dynamical_model])
        base_loss_fn_masked = partial(setup_lv.lv_base_loss_masked, state_dimension=state_dims[conf.dynamical_model])
    elif conf.dynamical_model == 'sir':
        # TODO @GiacomoFabrini find better way to express filepath and pass it as argument - same as above
        (train_data, t_save_train, t_span_train,
         validation_data, t_save_validation, t_span_validation) = setup_sir.get_training_validation_data()

        optx_config = {
            "rtol": 1e-6,  # changed from 1e-4
            "atol": 1e-6,  # changed from 1e-4
        }
        base_loss_fn = setup_sir.sir_base_loss
        base_loss_fn_masked = setup_sir.sir_base_loss_masked
    else:
        raise ValueError(f"Unknown dynamical model: {conf.dynamical_model}!")

    val_loss_fn = partial(compute_validation_loss, loss_fn=base_loss_fn)
    loss_fn = partial(compute_loss, loss_fn=base_loss_fn_masked)
    loss_grad_fn = partial(compute_loss_grad, loss_fn=base_loss_fn_masked)
    loss_optx = compute_loss_optx
    val_loss_optx = compute_val_loss_optx

    options = {
        "dynamic_params_scale": conf.dynamic_params_scale,
        "bounded_tanh_params": broadcast_bounds(
            bounds={"lb": conf.dynamic_params_lb, "ub": conf.dynamic_params_ub},
            shape=dynamic_params_shape
        )  # bounds are ignored if not using bounded tanh transform from Philipps et al. 2024
    }
    if conf.normalise_data:
        # Normalise initial condition u0 as well
        train_data, validation_data, u0, data_rescaling = normalise(train_data, validation_data, return_u0=True, u0=u0)
    else:
        data_rescaling = 1
    if conf.normalise_timepoints:
        # SaveAt
        t_save_train, t_save_validation, time_rescaling = normalise(t_save_train.subs.ts, t_save_validation.subs.ts)
        t_save_train, t_save_validation = SaveAt(ts=t_save_train), SaveAt(ts=t_save_validation)
        # Span
        t_span_train = [t_save_train.subs.ts[0], t_save_train.subs.ts[-1]]
        t_span_validation = [t_save_train.subs.ts[0], t_save_validation.subs.ts[-1]]
    else:
        time_rescaling = 1

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
    rescaling_args = {
        "data_rescaling": data_rescaling,
        "time_rescaling": time_rescaling,
    }

    # Pack arguments to return
    train_args = TrainValArgs(data=train_data, t_save=t_save_train, t_span=t_span_train, u0=u0_train)
    val_args = TrainValArgs(data=validation_data, t_save=t_save_validation, t_span=t_span_validation, u0=u0_val)
    loss_dict = {
        "base_loss_fn": base_loss_fn_masked,
        "loss_fn": loss_fn,
        "loss_grad_fn": loss_grad_fn,
        "loss_optx": loss_optx,
        "val_base_loss_fn": base_loss_fn,
        "val_loss_fn": val_loss_fn,
        "val_loss_optx": val_loss_optx,
    }
    option_args = optx_config, options
    return train_args, val_args, loss_dict, option_args, plot_args, rescaling_args


def broadcast_bounds(bounds, shape):
    return {
        key: tuple(jnp.broadcast_to(value, shape).tolist())
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
        ),
        conf.init_scale,
        key_init
    )
    return mlp


def generate_ms_time_and_mask(
        num_windows: int,
        shooting_window_length: int,
        training_horizon: int,
        t_save: SaveAt,
        make_plots: bool = False
) -> Tuple[Array, Array]:
    # Compute auxiliary MS variables
    t_windows, masks = [], []
    for i in range(num_windows):
        # Select the core timepoints
        start_idx = i * (shooting_window_length - 1)
        end_idx = (i + 1) * shooting_window_length - i
        core_indices = slice(start_idx, end_idx)
        core_values = t_save.subs.ts[core_indices]

        # Construct the full-length window
        window = np.zeros(training_horizon)
        window[: start_idx] = core_values[0]
        window[core_indices] = core_values
        window[end_idx:] = core_values[-1]
        t_windows.append(window)

        # Construct mask
        mask = np.zeros(training_horizon)
        mask[core_indices] = 1
        masks.append(mask)

    t_save_windows = jnp.array(t_windows)
    masks = jnp.array(masks)

    if make_plots:
        for arr, label in zip([t_save_windows, masks], ["time", "mask"]):
            plt.imshow(arr, vmin=0, vmax=1)
            ax = plt.gca()
            ax.yaxis.set_visible(False)  # Hide y-axis labels and ticks
            plt.savefig(plots_dir / f"multiple.{shooting_window_length}.noCL.{label}.pdf")
    return t_save_windows, masks


def setup_curriculum_learning(
        data: Array,
        t_save: Union[Array, SaveAt],
        n_epoch: int,
        shooting_mode: str,
        shooting_window_length: int, # only used for multiple shooting
        curriculum_learning: bool,
        num_curriculum_steps: int,
        max_curriculum_epochs: Optional[int],
        make_plots: bool = False
):
    if not curriculum_learning:
        return None
    else:
        # Check whether the shooting-mode is compatible with curriculum-learning
        if shooting_mode not in ["single", "multiple"]:
            raise ValueError(f"Curriculum learning cannot be used together with shooting_mode: {shooting_mode}")
        # Check the number of epochs to spend on curriculum learning is defined and sensible
        if max_curriculum_epochs is None:
            raise ValueError("Max curriculum steps must be defined for curriculum learning!")
        elif max_curriculum_epochs <= 0 or max_curriculum_epochs > n_epoch:
            raise ValueError(
                f"Max curriculum steps cannot be negative nor larger than number of epochs: {n_epoch}!")

        stage_dict = {}

        # Calculate the epoch at which to switch to full data
        switch_epoch = min(max_curriculum_epochs, n_epoch)

        if shooting_mode == "single":
            # Check the number of curriculum steps is defined and non-negative
            if num_curriculum_steps is None:
                raise ValueError("Number of steps must be defined for curriculum learning!")
            elif num_curriculum_steps <= 0:
                raise ValueError("Number of steps must be positive!")

            # Compute curriculum stage metrics
            for step in range(1, num_curriculum_steps + 1):
                length = int(len(data) * (step / num_curriculum_steps))
                start_epoch = int((step - 1) * (switch_epoch / num_curriculum_steps))
                end_epoch = int(step * (switch_epoch / num_curriculum_steps)) if step < num_curriculum_steps \
                    else n_epoch  # last CL stage lasts until final NN training epoch
                stage_dict[step] = {
                    "start": start_epoch,
                    "end": end_epoch,
                    "length": length,
                    "time": jnp.expand_dims(jnp.concatenate(
                        [t_save.subs.ts[:length], t_save.subs.ts[length-1]*jnp.ones(len(data)-length)]
                    ), axis=0),
                    "mask": jnp.expand_dims(jnp.zeros(len(data)).at[:length].set(1), axis=0),
                    "last_valid_index": 1, # only 1 valid window at each stage
                }
        elif shooting_mode == "multiple":
            # The number of curriculum steps is automatically defined by the number of shooting windows
            num_windows = int(np.ceil(len(data) / (shooting_window_length - 1)))
            print(f"Multiple shooting + curriculum learning selected. Number of curriculum steps will be "
                  f"overridden with {num_windows}.")
            new_num_curriculum_steps = num_windows
            # Fill the incremental data list with the incremental data
            for step in range(1, new_num_curriculum_steps + 1):
                start_epoch = int((step - 1) * (switch_epoch / (new_num_curriculum_steps-1)))
                end_epoch = int(
                    step * (switch_epoch / (new_num_curriculum_steps-1))
                ) if step < new_num_curriculum_steps else n_epoch  # last CL stage lasts until final NN training epoch
                if step == 1:
                    time, mask = generate_ms_time_and_mask(
                        num_windows, shooting_window_length, len(data), t_save
                    )
                else:
                    # Subset to non-zero masks and time to compute next CL stage ones
                    if step > 2:
                        mask = mask[:-(step-2)]
                        time = time[:-(step-2)]
                    # Merge adjacent windows and replace the last (disappearing) one with a zero mask
                    merged_mask = jnp.concatenate([
                        jnp.logical_or(mask[:-1], mask[1:]).astype(float),
                        jnp.zeros((step-1, len(data)))
                    ], axis=0)
                    # Update time values
                    merged_time = jnp.where(mask[:-1] == 1, time[:-1], time[1:])
                    # Fix mis-padded values by propagating the lowest valid timepoint forward
                    min_valid_timepoints = jnp.argmin(merged_time, axis=1, keepdims=True)
                    min_valid_time = jnp.min(merged_time, axis=1, keepdims=True)
                    indices = jnp.arange(merged_time.shape[1])[None, :]  # Shape (1, num_timepoints) for broadcasting
                    override_mask = indices < min_valid_timepoints  # Shape (num_windows, num_timepoints)
                    merged_time = jnp.where(override_mask, min_valid_time, merged_time)
                    # and replace the last window with zeros
                    merged_time = jnp.concatenate([
                        merged_time, jnp.zeros((step-1, len(data)))
                    ], axis=0)

                    mask = merged_mask
                    time = merged_time

                stage_dict[step] = {
                    "start": start_epoch,
                    "end": end_epoch,
                    "length": mask[0].sum(),
                    "time": time,
                    "mask": mask.astype(int),
                    "last_valid_index": num_windows - (step-1)
                }

        if make_plots:
            for attribute in ["time", "mask", "mask_overlap"]:
                plt.subplots(len(stage_dict.keys()), 1, figsize=(10, 3*len(stage_dict.keys())))
                for stage in sorted(stage_dict.keys()):
                    plt.subplot(len(stage_dict.keys()), 1, stage)
                    if attribute in ["time", "mask"]:
                        if len(jnp.array(stage_dict[stage][attribute]).shape) == 1:
                            arr = jnp.expand_dims(jnp.array(stage_dict[stage][attribute]), axis=0)
                        else:
                            arr = jnp.array(stage_dict[stage][attribute])
                    else:
                        if len(jnp.array(stage_dict[stage]["mask"]).shape) == 1:
                            arr = jnp.expand_dims(
                                jnp.array(stage_dict[stage]["mask"])[:-1] & jnp.array(stage_dict[stage]["mask"])[1:],
                                axis=0
                            )
                        else:
                            arr = jnp.array(stage_dict[stage]["mask"])[:-1] & jnp.array(stage_dict[stage]["mask"])[1:]
                    plt.imshow(arr, vmin=0, vmax=1, aspect="auto")
                    ax = plt.gca()
                    ax.yaxis.set_visible(False)  # Hide y-axis labels and ticks
                    plt.title(f"CL Stage {stage}")
                plt.tight_layout()
                plt.savefig(
                    plots_dir / f"{shooting_mode}.{shooting_window_length}.{curriculum_learning}.{attribute}.pdf"
                )
        return stage_dict


def initialize_single_shooting(
    training_horizon: int,
    t_save: Union[Array, SaveAt],
    u0_guess: Tuple,
    curriculum_learning: bool,
    curriculum_params: Optional[dict[str, tuple]] = None
) -> tuple[Array, Array, Array]:
    if curriculum_learning:
        if curriculum_params is not None:
            t_windows = curriculum_params["time"]
            masks = curriculum_params["mask"]
        else:
            raise ValueError("curriculum_indices must be provided if curriculum_learning is True")
    else:
        t_windows = jnp.array([t_save.subs.ts])
        masks = jnp.array([jnp.ones(training_horizon)])
    u0 = jnp.expand_dims(jnp.array(u0_guess), axis=0)
    return t_windows, masks, u0


def initialize_multiple_shooting(
        effective_state_dimension: int,
        shooting_window_length: int,
        training_horizon: int,
        t_save: Union[Array, SaveAt],
        strategy: str,
        u0_guess: Optional[Tuple],
        curriculum_learning: bool = False,
        curriculum_params: Optional[dict[str, tuple]] = None,
        random_seed: int = 0,
) -> tuple[Array, Array, Array]:
    """
    Initialize auxiliary variables (num_windows, starting and ending points,
    save timepoints) and learnable initial values for the shooting windows.

    Parameters
    ----------
    effective_state_dimension: int
        The number of input features for the neural network. Includes state variables and augmentation variables.
    shooting_window_length: int
        The number of timepoints in each shooting window.
    training_horizon: int
        Overall training horizon (time-range of training datapoints).
    t_save: diffrax.SaveAt
        Timepoints at which to save the training simulation.
    strategy: str
        Initialization strategy for initial conditions.
    u0_guess: Tuple
        Initial guess for the initial conditions.
    curriculum_learning: bool
        If True, the model will be trained using curriculum learning.
    curriculum_params: dict
        Timepoints and masks for first curriculum learning stage.
    random_seed: int
        Random seed for random initialization.
    """
    # Pre-compute the number of windows
    num_windows = int(np.ceil(training_horizon / (shooting_window_length - 1)))

    if curriculum_learning and (curriculum_params is not None):
        # pre-computed
        t_save_windows, masks = curriculum_params["time"], curriculum_params["mask"]
    elif curriculum_learning and (curriculum_params is None):
        raise ValueError("curriculum_params must be provided if curriculum_learning is True!")
    else:
        t_save_windows, masks = generate_ms_time_and_mask(
            num_windows, shooting_window_length, training_horizon, t_save
        )

    if strategy == 'z':
        return t_save_windows, masks, jnp.zeros((num_windows, effective_state_dimension))
    # TODO @GiacomoFabrini add an option to initialise with observed initial conditions
    elif strategy == 'g':
        return t_save_windows, masks, jnp.tile(u0_guess, (num_windows, 1))
    elif strategy == 'ru':
        # Sample uniformly state_dimension variables
        u0 = jr.uniform(jr.PRNGKey(random_seed), (num_windows, effective_state_dimension))
        return t_save_windows, masks, u0
    else:
        raise ValueError(f"Invalid shooting_iv_init strategy: {strategy}")


def setup_time_masks_u0_curriculum(
        conf: Conf,
        train_args: TrainValArgs,
        u0_guess: Union[tuple, jnp.ndarray],
        effective_state_dimension: int,
) -> Tuple[Array, Array, Array, dict]:
    # Setup curriculum learning (if enabled)
    # TODO this crashes the run - might want to think about a different approach or is this fine?
    if conf.curriculum_learning and conf.num_epochs_adam == 0:
        raise ValueError(
            f"Number of NN training epochs must be > 0 to use curriculum learning - found {conf.num_epochs_adam}!"
        )
    curriculum_stage_dict = setup_curriculum_learning(
        data=train_args.data,
        t_save=train_args.t_save,
        n_epoch=conf.num_epochs_adam,
        shooting_mode=conf.shooting_mode,
        shooting_window_length=conf.shooting_window_length,
        curriculum_learning=conf.curriculum_learning,
        num_curriculum_steps=conf.num_curriculum_steps,
        max_curriculum_epochs=conf.max_curriculum_epochs,
    )
    if conf.shooting_mode == "single":
        t_save_windows, masks, u0 = initialize_single_shooting(
            training_horizon=train_args.data.shape[0],
            t_save=train_args.t_save,
            u0_guess=u0_guess,
            curriculum_learning=conf.curriculum_learning,
            curriculum_params=curriculum_stage_dict[1] if conf.curriculum_learning else None
        )
    elif conf.shooting_mode == "multiple":
        t_save_windows, masks, u0 = initialize_multiple_shooting(
            effective_state_dimension=effective_state_dimension,
            shooting_window_length=conf.shooting_window_length,
            training_horizon=train_args.data.shape[0],
            t_save=train_args.t_save,
            strategy=conf.shooting_iv_init,
            u0_guess=u0_guess,
            curriculum_learning=conf.curriculum_learning,
            curriculum_params=curriculum_stage_dict[1] if conf.curriculum_learning else None,
            random_seed=conf.multistart_job,
        )
    else:
        raise ValueError(f"Unknown shooting mode: {conf.shooting_mode}!")
    return t_save_windows, masks, u0, curriculum_stage_dict


def load_multistart_dynamic_params(conf: Conf, data_dir: Path) -> jnp.ndarray:
    """
    Load pre-sampled log10 dynamic parameters from CSV and apply correct scale.

    Returns:
        jnp.ndarray of shape (n_params,) in the correct scale (log10, linear, or bounded_tanh)
    """
    # Locate the correct file
    multistart_file = (
        data_dir / conf.dynamical_model /
        f"{conf.dynamic_params_perturb_method}__{conf.multistart_deviation}.csv"
    )

    # Load the multistart CSV
    df = pd.read_csv(multistart_file)

    # Select the correct row by job index
    row = df[df["multistart_job"] == conf.multistart_job]
    if row.empty:
        raise ValueError(f"No row found for multistart_job={conf.multistart_job} in {multistart_file}")

    # Extract log10 values
    log_params = row.filter(like="param_").values.flatten()
    true_shape = true_dynamic_parameters[conf.dynamical_model].shape
    if log_params.shape[0] != true_shape[0]:
        raise ValueError(f"Expected {true_shape[0]} parameters but got {log_params.shape[0]}")

    # Convert to required scale
    if conf.dynamic_params_scale == "log10":
        return jnp.array(log_params)
    elif conf.dynamic_params_scale == "linear":
        return jnp.power(10, jnp.array(log_params))
    elif conf.dynamic_params_scale == "bounded_tanh":
        return bounded_tanh_transform_params(
            params=jnp.power(10, jnp.array(log_params)),
            **broadcast_bounds(
                bounds={
                    "lb": conf.dynamic_params_lb,
                    "ub": conf.dynamic_params_ub
                },
                shape=true_dynamic_parameters[conf.dynamical_model].shape
            )
        )
    else:
        raise ValueError(f"Unsupported dynamic_params_scale: {conf.dynamic_params_scale}")
