import diffrax
import equinox as eqx
import itertools as itt
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jaxtyping import Array
from src.common import plots_dir, Conf
from src.model_training.training_util import generate_incremental_arrays
from src.model_training.losses import compute_loss
from src.ude_problem import UDEProblem, CustomDropoutMLP
from typing import Dict, List, Optional, Union
from tqdm.contrib.itertools import product  # directly using itertools.product progress bar wrapper

def analyse_lv_loss_landscape_dynparams(
        udeprob: UDEProblem,
        u0: jnp.ndarray,
        train_data: jnp.ndarray,
        t_save_train: diffrax.SaveAt,
        t_span_train: list,
        validation_data: jnp.ndarray,
        t_save_validation: diffrax.SaveAt,
        t_span_validation: list,
        conf: Conf,
        N_EPOCHS: int,
        opt_dynamic_params: Dict[int, dict[str, Array]],
        do_plots: bool,
        fig_name: str,
        fig_fmts: Union[tuple[str], list[str]] = ("pdf", "svg"),
        num_points: int = 200,
):
    # We treat CL/no CL in the same way - no CL simply has a single fictitious CL step
    # Generate incremental training data for NN training - curriculum learning or not (if not, full data at each epoch)
    incremental_training_data, incremental_train_save, incremental_train_span, steps = generate_incremental_arrays(
        data=train_data,
        t_save=t_save_train,
        n_epoch=N_EPOCHS,
        curriculum_learning=conf.curriculum_learning,
        num_curriculum_steps=conf.num_curriculum_steps,
        max_curriculum_epochs=conf.max_curriculum_epochs
    )
    num_curriculum_steps = conf.num_curriculum_steps

    # Override steps and num_curriculum_steps if not using curriculum learning
    if not conf.curriculum_learning:
        steps = [0]  # fictitious step for non-curriculum learning
        num_curriculum_steps = 1

    # Set centre of the grid to the dynamic parameters of the found minimiser -- work in log-scale
    if conf.dynamic_params_scale == "log10":
        log_alpha, log_delta = udeprob.func.dynamic_parameters.tolist()
    else:
        log_alpha, log_delta = np.log10(udeprob.func.dynamic_parameters).tolist()

    # Define scan ranges for alpha and delta spanning sampling range
    alpha_scan_range, delta_scan_range = [
        np.linspace(
            log_param - conf.multistart_deviation,
            log_param + conf.multistart_deviation,
            num=num_points
        )
        for log_param in [log_alpha, log_delta]
    ]

    dfs = []

    for alpha_test, delta_test in product(alpha_scan_range, delta_scan_range, position=0, leave=True):
        # Replace dynamic parameters and partition
        new_udeprob = udeprob.copy(
            new_dynamic_parameters=jnp.array([alpha_test, delta_test]),
            new_options={"dynamic_params_scale": "log10"}  # ensure parameters are on log10 scale since we are initialising in log scale
        )
        params, static = eqx.partition(new_udeprob, eqx.is_array)

        # compute validation loss one time only
        loss_val, _ = compute_loss(
            prob_params=params, prob_static=static,
            u0=jnp.concatenate([u0, jnp.zeros(1)]) if new_udeprob.func.regularise_network else u0,
            t_save=t_save_validation,
            t_span=t_span_validation,
            data=validation_data,
            mlp_reg_exp=conf.mlp_reg_exp, mlp_reg_strength=conf.mlp_reg_strength,
            output_reg=conf.output_reg,
        )
        # compute training loss once per curriculum stage
        for curriculum_step in range(num_curriculum_steps):
            loss_train, _ = compute_loss(
                prob_params=params, prob_static=static,
                u0=jnp.concatenate([u0, jnp.zeros(1)]) if new_udeprob.func.regularise_network else u0,
                t_save=incremental_train_save[steps[curriculum_step]],
                t_span=incremental_train_span[steps[curriculum_step]],
                data=incremental_training_data[steps[curriculum_step]],
                mlp_reg_exp=conf.mlp_reg_exp, mlp_reg_strength=conf.mlp_reg_strength,
                output_reg=conf.output_reg,
            )
            dfs.append(
                pd.DataFrame(
                    {
                        "alpha": [alpha_test],
                        "delta": [delta_test],
                        "loss_train": [loss_train],
                        "loss_val": [loss_val],
                        "curriculum_step": [curriculum_step]
                    }
                )
            )

    # Concatenate resulting dataframes and convert loss values to floats
    df = pd.concat(dfs)
    df["loss_train"] = df["loss_train"].astype(float)
    df["loss_val"] = df["loss_val"].astype(float)

    # ================================================================
    #                            PLOTTING
    # ================================================================
    if do_plots:
        alpha_values = df['alpha'].unique()
        delta_values = df['delta'].unique()
        alpha_mesh, delta_mesh = np.meshgrid(alpha_values, delta_values)

        if num_curriculum_steps <= 5:
            subplot_rows=1
            subplot_cols=num_curriculum_steps
        elif num_curriculum_steps % 2 ==0:
            subplot_rows=2
            subplot_cols=num_curriculum_steps//2
        else:
            subplot_rows=2
            subplot_cols=num_curriculum_steps//2+1
        plt.subplots(subplot_rows, subplot_cols, figsize=(10 * subplot_cols, 6 * subplot_rows))
        for curriculum_step in df.curriculum_step.unique():
            # Reshaping loss_train values to fit the meshgrid
            loss_mesh = np.log10(
                df[df.curriculum_step==curriculum_step].pivot(
                    index='delta', columns='alpha', values=f'loss_train'
                ).values
            )

            # Creating the contour plot
            plt.subplot(subplot_rows, subplot_cols, curriculum_step + 1)
            # Draw contour lines
            plt.contour(alpha_mesh, delta_mesh, loss_mesh, levels=100, alpha=0.1)
            # Fill contour
            contour = plt.contourf(alpha_mesh, delta_mesh, loss_mesh, levels=100, cmap='coolwarm')
            # Add dynamic params for start/end of CL step in black/magenta stars, respectively
            for timepoint, color in zip(["start", "end"], ["black", "magenta"]):
                if timepoint in opt_dynamic_params[curriculum_step].keys():
                    plt.scatter(
                        opt_dynamic_params[curriculum_step][timepoint][0],
                        opt_dynamic_params[curriculum_step][timepoint][1],
                        marker='*',
                        c=color,
                        s=300
                    )
            # Add reference lines
            plt.axvline(log_alpha, c='gray', ls='--')
            plt.axhline(log_delta, c='gray', ls='--')
            plt.colorbar(contour)
            plt.xlabel('Log10 Alpha')
            plt.ylabel('Log10 Delta')
            plt.title(f'Loss train - {curriculum_step}')
        for fig_fmt in fig_fmts:
            plt.savefig(plots_dir / f"{fig_name}.{fig_fmt}")
        plt.show()
    return df


def get_params(model: Union[eqx.nn.MLP, CustomDropoutMLP, UDEProblem]):
    """Extract parameters from an equinox.nn.MLP model and return a list of tensors."""
    # The parameters in an Equinox model can be accessed through `eqx.tree_flatten`
    params = jax.tree_util.tree_flatten(model)[0]
    return [param for param in params if isinstance(param, Array)]


def get_random_params(params, key):
    """
    Produce a random direction that is a list of random Gaussian tensors/arrays
    with the same shape as the network's weights & biases (params), so one direction entry per parameter.

    Parameters:
    - params: list of JAX arrays representing the network's parameters.
    - key: JAX PRNGKey for random number generation.

    Returns:
    - A list of JAX arrays with random Gaussian values and the same shapes as the parameters.
    """
    # Create a new random key for each weight tensor
    keys = jax.random.split(key, len(params))
    random_directions = [jax.random.normal(k, shape=p.shape) for k, p in zip(keys, params)]
    return random_directions


def normalize_direction(direction, weights, norm='filter'):
    """
    Rescale the direction so that it has a similar norm as the corresponding
    weights in different levels.

    Args:
      direction: list of JAX arrays, random directions for one layer
      weights: list of JAX arrays, original model weights for one layer
      norm: normalization method, 'filter' | 'layer' | 'weight' | 'dfilter' | 'dlayer'

    Returns:
      List of JAX arrays with normalized direction values.
    """
    if norm == 'filter':
        # Rescale each filter so that it has the same norm as the corresponding filter in weights
        normalized_direction = [
            d * (jnp.linalg.norm(w) / (jnp.linalg.norm(d) + 1e-10)) for d, w in zip(direction, weights)
        ]
    elif norm == 'layer':
        # Rescale the entire layer to match the norm of the layer weights
        layer_norm_factor = jnp.linalg.norm(jnp.concatenate([w.ravel() for w in weights]))
        direction_norm = jnp.linalg.norm(jnp.concatenate([d.ravel() for d in direction]))
        normalized_direction = [d * (layer_norm_factor / (direction_norm + 1e-10)) for d in direction]
    elif norm == 'weight':
        # Scale each direction entry by the corresponding weight
        normalized_direction = [d * w for d, w in zip(direction, weights)]
    elif norm == 'dfilter':
        # Scale each filter direction to unit norm
        normalized_direction = [d / (jnp.linalg.norm(d) + 1e-10) for d in direction]
    elif norm == 'dlayer':
        # Scale the entire layer direction to unit norm
        direction_norm = jnp.linalg.norm(jnp.concatenate([d.ravel() for d in direction]))
        normalized_direction = [d / (direction_norm + 1e-10) for d in direction]
    else:
        raise ValueError(f"Unsupported norm type: {norm}")

    return normalized_direction


def normalize_directions_for_weights(direction, weights, norm='filter', ignore='biasbn'):
    """
    Normalize direction entries according to the weights.

    Args:
      direction: list of JAX arrays, random directions for each layer
      weights: list of JAX arrays, original weights for each layer
      norm: normalization method, 'filter' | 'layer' | 'weight' | 'dfilter' | 'dlayer'
      ignore: if 'biasbn', ignores directions with dimensions <= 1

    Returns:
      List of JAX arrays with normalized directions.
    """
    assert len(direction) == len(weights)
    normalized_directions = []

    for d, w in zip(direction, weights):
        if d.ndim <= 1:
            if ignore == 'biasbn':
                # Ignore directions for biases (params with 1 dimension)
                normalized_directions.append(jnp.zeros_like(d))
            else:
                normalized_directions.append(w)
        else:
            # Normalize using the specified method (default = "filter")
            normalized_directions.append(normalize_direction([d], [w], norm=norm)[0])

    return normalized_directions

def create_random_direction(
        model: UDEProblem,
        key: Array,
        ignore='biasbn',
        norm='filter'
):
    """
        Set up a random (normalized) direction with the same dimension as
        the weights or states.

        Args:
          model: the given trained UDEProblem model
          key: PRNG key for random number generation.
          ignore: 'biasbn', ignore biases and BN parameters.
          norm: direction normalization method, including
                'filter' | 'layer' | 'weight' | 'dlayer' | 'dfilter'

        Returns:
          direction: a random direction with the same dimension as weights or states.
    """

    params = get_params(model) # a list of parameters.
    direction = get_random_params(params, key)
    nromalised_direction = normalize_directions_for_weights(direction, params, norm, ignore)
    return nromalised_direction


def set_weights(
        model: UDEProblem,
        directions: Optional[List[List[jnp.ndarray]]] = None,
        step: Optional[Union[float, List[float]]] = None,
        method: str = "UDE"  # UDE or MLP
) -> Union[eqx.nn.MLP, CustomDropoutMLP, UDEProblem]:
    """
    Create a new model instance with parameters shifted in the given direction by a specified step.

    Args:
      model: The original model instance.
      directions: Optional list of lists of JAX arrays representing directions for each parameter.
      step: Optional step size to scale the directions. If directions are provided, step must also be provided.
      method: string describing which parameters to shift: UDE - full UDE model; MLP - MLP model only

    Returns:
      A new model instance with updated weights.
    """

    if method == "UDE":
        params = get_params(model)
    elif method == "MLP":
        params = get_params(model.func.mlp)
        directions = [direction[1:] for direction in directions]  # discard direction associated to dynamic parameters
    else:
        raise ValueError(f"Unknown method: {method}")

    # If directions are not provided, we simply set the weights as specified
    if directions is None:
        new_params = params
    else:
        assert step is not None, 'If a direction is specified then step must be specified as well'

        # Calculate the change along the specified direction and step size
        if len(directions) == 2:
            # When two direction sets are given, apply both with corresponding steps
            dx, dy = directions
            changes = [d0 * step[0] + d1 * step[1] for d0, d1 in zip(dx, dy)]
        else:
            # When a single direction set is given
            changes = [d * step for d in directions[0]]

        # Apply changes to the weights
        new_params = [w + d for w, d in zip(params, changes)]

    if method == "UDE":
        new_mlp_params = new_params[1:]  # discard dynamic parameter update for now
    else:
        new_mlp_params = new_params

    # Get a PyTreeDef for the MLP
    _, mlp_model_tree_def = jax.tree_util.tree_flatten(model.func.mlp)
    # Add back activations to MLP new mlp params
    new_mlp_params.extend([model.func.mlp.activation, model.func.mlp.final_activation])
    # and unflatten into functional updated MLP
    new_mlp_model = jax.tree_util.tree_unflatten(mlp_model_tree_def, new_mlp_params)

    # If model was a UDEProblem, need to replace dynamic parameters and the MLP
    if method == "UDE":
        # Replace with shifted MLP as well as shifted dynamic parameters
        new_model = model.replace_mlp(
            new_mlp=new_mlp_model,
        )
        new_model = new_model.replace_dynamic_parameters(
            new_dynamic_parameters=new_params[0],
        )
    elif method == "MLP":
        # Only replace MLP
        new_model = model.replace_mlp(
            new_mlp=new_mlp_model,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    return new_model


def get_shifted_model(
        model: UDEProblem,
        key,
        num_dirs: int,
        step_sizes: List[float],
        method: str = "UDE",
        normalisation_ignore="biasbn",
):
    # Split PRNG key
    dirkeys = jax.random.split(key, num_dirs)
    # Generate directions
    directions = []
    for _, dirkey in enumerate(dirkeys):
        # Generate normalised directions and append to model
        directions.append(create_random_direction(model, dirkey, ignore=normalisation_ignore))

    shifted_model = set_weights(
        model=model,
        directions=directions,
        step=step_sizes,
        method=method,
    )

    return shifted_model

def get_ude_param_landscape(
    udeprob: UDEProblem,
    u0: jnp.ndarray,
    train_data: jnp.ndarray,
    t_save_train: diffrax.SaveAt,
    t_span_train: list,
    validation_data: jnp.ndarray,
    t_save_validation: diffrax.SaveAt,
    t_span_validation: list,
    conf: Conf,
    key_list: Union[Array, list[Array]],
    num_dirs: int,
    start_stop: List[float],
    num_points: int,
    which_params: List[str],
    do_plots: bool,
    fig_name: str,
    fig_fmts: Union[tuple[str], list[str]] = ("pdf", "svg"),
):
    # Unpack start and stop points
    start, stop = start_stop
    # Define scanning ranges for random directions
    step = (stop - start) / num_points
    scan_range = jnp.arange(start, stop + step, step)

    if isinstance(key_list, Array):
        key_list = [key_list]

    dfs = []
    for i, key in enumerate(key_list):
        # Split PRNG key
        dirkeys = jax.random.split(key, num_dirs)
        # Generate directions
        directions = []
        for _, dirkey in enumerate(dirkeys):
            # Generate normalised directions and append to model
            directions.append(create_random_direction(udeprob, dirkey, ignore="False"))

        for step1, step2, method in product(scan_range, scan_range, which_params, position=0, leave=True):
            shifted_model = set_weights(
                model=udeprob,
                directions=directions,  # use precomputed directions (kept constant while scanning)
                step=[step1, step2],
                method=method,
            )

            prob_params, prob_static = eqx.partition(shifted_model, eqx.is_array)

            # TODO @GiacomoFabrini -- need to fix with new loss function logic for multiple-shooting, etc.
            loss_train, _ = compute_loss(
                prob_params=prob_params,
                prob_static=prob_static,
                u0=u0 if not shifted_model.func.regularise_network
                else jnp.concatenate([u0, jnp.zeros(1)]),
                t_save=t_save_train,
                t_span=np.array(t_span_train, dtype=np.float64),
                data=train_data,
                mlp_reg_exp=conf.mlp_reg_exp,
                mlp_reg_strength=conf.mlp_reg_strength,
                output_reg=conf.output_reg,
            )

            loss_val, _ = compute_loss(
                prob_params=prob_params,
                prob_static=prob_static,
                u0=u0 if not shifted_model.func.regularise_network
                else jnp.concatenate([u0, jnp.zeros(1)]),
                t_save=t_save_validation,
                t_span=np.array(t_span_validation, dtype=np.float64),
                data=validation_data,
                mlp_reg_exp=conf.mlp_reg_exp,
                mlp_reg_strength=conf.mlp_reg_strength,
                output_reg=conf.output_reg,
            )

            dfs.append(
                pd.DataFrame(
                    {
                        "x": [step1],
                        "y": [step2],
                        "loss_train": [loss_train],
                        "loss_val": [loss_val],
                        "method": [method],
                        "num_random_dir": [i],
                    }
                )
            )

    df = pd.concat(dfs)
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    df["loss_train"] = df["loss_train"].astype(float)
    df["loss_val"] = df["loss_val"].astype(float)

    if do_plots:
        # Create a meshgrid and calculate loss on mesh
        x_values = df['x'].unique()
        y_values = df['y'].unique()
        x_mesh, y_mesh = np.meshgrid(x_values, y_values)

        for dataset, method in itt.product(["train", "val"], which_params):
            loss_mesh = np.log10(
                df[df.method==method].pivot(index='y', columns='x', values=f'loss_{dataset}').values
            )
            # Draw contour lines
            plt.contour(x_mesh, y_mesh, loss_mesh, levels=100, alpha=0.1)
            # Fill contour
            contour = plt.contourf(x_mesh, y_mesh, loss_mesh, levels=100, cmap='coolwarm')
            # Add found minimiser
            plt.scatter(0, 0, c='red', s=5)
            plt.colorbar(contour)
            plt.xlabel('Step1')
            plt.ylabel('Step2')
            plt.title(f'Contour plot of loss {dataset} along normalised random directions')
            for fig_fmt in fig_fmts:
                plt.savefig(plots_dir / f"{fig_name}_{dataset}_{method}.{fig_fmt}")
            plt.show()
    return df