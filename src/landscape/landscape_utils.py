import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jaxtyping import Array
# from matplotlib.colors import LogNorm
from scipy.stats import qmc
from sklearn.decomposition import PCA
from src.model_training.model_util import process_mechanistic_params
from src.model_training.training_util import update_hessian_index_mask_multiple_curriculum
from src.ude_problem import UDEProblem, CustomDropoutMLP
from tqdm import tqdm
from typing import Callable, Optional, Union


def flatten_tensor_list(tensor_list: list[Array]) -> tuple[Array, list[tuple]]:
    flat_list, metadata = [], []
    for array in tensor_list:
        flat_list.append(array.ravel())
        metadata.append((array.shape, array.size))
    return jnp.concatenate(flat_list), metadata


def unflatten_tensor_list(flat_vector: Array, metadata: list[tuple]) -> list[Array]:
    tensor_list, idx = [], 0
    for shape, size in metadata:
        tensor = flat_vector[idx:idx + size].reshape(shape)
        tensor_list.append(tensor)
        idx += size
    return tensor_list


def get_params(model: Union[eqx.nn.MLP, CustomDropoutMLP, UDEProblem]) -> list[Array]:
    """Extract parameters from an equinox.nn.MLP model and return a list of tensors."""
    params = jax.tree_util.tree_flatten(model)[0]
    return [param for param in params if isinstance(param, Array)]


def get_params_conditional(model: UDEProblem, which: str = "MLP") -> list[Array]:
    if which == "MLP":
        return get_params(model.func.mlp)
    elif which == "dynamic":
        return model.func.dynamic_parameters.tolist()
    else:
        return [model.func.dynamic_parameters, *get_params(model.func.mlp)]


def get_diff_weights(weights1: list[Array], weights2: list[Array]) -> list[Array]:
    return [w2 - w1 for w1, w2 in zip(weights1, weights2)]


def project_1d(w: Array, d: Array) -> float:
    return (jnp.dot(w, d) / jnp.linalg.norm(d)).item()


def project_2d(
    w: Array,
    dx: Union[Array, np.ndarray],
    dy: Union[Array, np.ndarray]
) -> tuple[float, float]:
    return project_1d(w, dx), project_1d(w, dy)


def setup_weight_pca_directions(
    model_dict,
    last_stage,
    last_weights,
    last_dynamic,
    which="MLP"
):
    diff_matrix = []
    for stage in sorted(model_dict):
        if stage == last_stage:
            continue
        model = model_dict[stage]
        current = get_params_conditional(model, which)
        last = last_dynamic if which == "dynamic" \
            else last_weights if which == "MLP" \
            else [last_dynamic, *last_weights]
        diff_vector = get_diff_weights(last, current)
        flat_diff, _ = flatten_tensor_list(diff_vector)
        diff_matrix.append(flat_diff)

    pca = PCA(n_components=2)
    pca.fit(np.array(diff_matrix))
    pc1, pc2 = map(np.array, pca.components_[:2])

    print(f"PCA cosine angle: {jnp.dot(pc1, pc2) / (jnp.linalg.norm(pc1) * jnp.linalg.norm(pc2))}")
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

    return diff_matrix, pc1, pc2


def project_trajectory(model_dict, which="MLP"):
    last_stage = max(model_dict)
    last_model = model_dict[last_stage]
    last_weights = get_params(last_model.func.mlp)
    last_dynamic = last_model.func.dynamic_parameters

    matrix, pc1, pc2 = setup_weight_pca_directions(
        model_dict, last_stage, last_weights, last_dynamic, which=which
    )

    coord_dfs = []
    for stage, vec in zip(sorted(model_dict), matrix):
        if stage == last_stage:
            continue
        x, y = project_2d(vec, pc1, pc2)
        coord_dfs.append(pd.DataFrame({'x': [x], 'y': [y], 'stage': [stage]}))

    return (
        pd.concat(coord_dfs, ignore_index=True),
        ([df['x'][0] for df in coord_dfs], [df['y'][0] for df in coord_dfs]),
        (pc1, pc2)
    )


def rebuild_mlp_from_weights(
    template: eqx.nn.MLP,
    weights: list[Array]
) -> eqx.nn.MLP:
    new_layers, idx = [], 0
    for layer in template.layers:
        if hasattr(layer, 'bias'):
            new_layer = eqx.tree_at(
                lambda l: (l.weight, l.bias),
                layer, (weights[idx], weights[idx + 1])
            )
            idx += 2
        else:
            new_layer = eqx.tree_at(
                lambda l: l.weight,
                layer, weights[idx]
            )
            idx += 1
        new_layers.append(new_layer)

    return eqx.tree_at(lambda m: m.layers, template, tuple(new_layers))


def update_model_from_params(model, new_params, which: str):
    if which == "MLP":
        return eqx.tree_at(
            lambda m: m.func.mlp, model,
            rebuild_mlp_from_weights(model.func.mlp, new_params)
        )
    elif which == "dynamic":
        return eqx.tree_at(
            lambda m: m.func.dynamic_parameters, model, jnp.array(new_params)
        )
    else:
        dyn, mlp_weights = new_params[0], new_params[1:]
        updated = eqx.tree_at(lambda m: m.func.dynamic_parameters, model, dyn)
        return eqx.tree_at(
            lambda m: m.func.mlp, updated,
            rebuild_mlp_from_weights(model.func.mlp, mlp_weights)
        )


def compute_hessian_directions(
    model,
    conf,
    t_save,
    t_span,
    data,
    last_valid_index,
    base_loss_fn,
    loss_hessian_fn,
    active_idx = None,
) -> tuple[Array, Array]:
    params, static = eqx.partition(model, eqx.is_array)
    flat_p, unflatten_p = jax.flatten_util.ravel_pytree(params)
    # Compute hessian
    hessian = loss_hessian_fn(
        flat_p, unflatten_p, static,
        t_save, t_span, data, last_valid_index,
        conf.mlp_reg_exp, conf.mlp_reg_strength,
        conf.output_reg,
        conf.multiple_shooting_penalty,
        conf.shooting_mode,
        base_loss_fn
    )
    if conf.shooting_mode == "multiple" and conf.curriculum_learning:
        # Only use the active hessian indices (remove initial values for output regularisation and dropped windows)
        unfrozen_hessian = hessian[jnp.ix_(active_idx, active_idx)]
        eigvals, unfrozen_eigvecs = jnp.linalg.eigh(unfrozen_hessian)
        # Pad eigenvectors back to full size with zeros
        n_active = unfrozen_eigvecs.shape[0]
        eigvecs = jnp.zeros((hessian.shape[0], n_active), dtype=unfrozen_eigvecs.dtype)
        eigvecs = eigvecs.at[active_idx, :].set(unfrozen_eigvecs)
    else:
        eigvals, eigvecs = jnp.linalg.eigh(hessian)

    # Select the two dominant eigenvalues and the corresponding eigenvectors
    top_indices = jnp.argsort(jnp.abs(eigvals))[-2:]
    dir1, dir2 = eigvecs[:, top_indices[0]], eigvecs[:, top_indices[1]]
    # Exclude directions corresponding to initial_values
    num_excluded = model.initial_values.shape[0] * model.initial_values.shape[1]
    return dir1[:-num_excluded], dir2[:-num_excluded]


def evaluate_loss_on_mesh_grid(
    xmesh,
    ymesh,
    base_flat_params,
    metadata,
    model,
    loss_fn,
    conf,
    data,
    t_save,
    masks,
    last_valid_index,
    dir1,
    dir2,
    which
):
    losses = []
    for x, y in zip(xmesh.ravel(), ymesh.ravel()):
        offset = x * dir1 + y * dir2
        new_flat = base_flat_params + offset
        new_params = unflatten_tensor_list(new_flat, metadata)
        new_model = update_model_from_params(model, new_params, which)
        params, static = eqx.partition(new_model, eqx.is_array)

        loss, _ = loss_fn(
            params,
            static,
            t_save,
            masks,
            data,
            last_valid_index,
            conf.mlp_reg_exp,
            conf.mlp_reg_strength,
            conf.output_reg,
            conf.multiple_shooting_penalty,
            conf.shooting_mode
        )
        losses.append(loss)
    return np.array(losses)


def evaluate_loss_on_pca_grid(
    model,
    pc1,
    pc2,
    xcoords,
    ycoords,
    loss_fn,
    conf,
    data,
    t_save,
    masks,
    last_valid_index,
    which="MLP",
    num_points=20
):
    flat_params, metadata = flatten_tensor_list(get_params_conditional(model, which))
    xmesh, ymesh = np.meshgrid(
        np.linspace(min(xcoords), max(xcoords), num_points),
        np.linspace(min(ycoords), max(ycoords), num_points)
    )
    return evaluate_loss_on_mesh_grid(
        xmesh,
        ymesh,
        flat_params,
        metadata,
        model,
        loss_fn,
        conf,
        data,
        t_save,
        masks,
        last_valid_index,
        pc1,
        pc2,
        which
    )


def evaluate_loss_random_directions(
    model1,
    loss_fn,
    conf,
    data,
    t_save,
    masks,
    last_valid_index,
    which="MLP",
    num_points=20,
    key=jr.PRNGKey(0),
    normalise=True,
    model2=None
):
    f1, meta = flatten_tensor_list(get_params_conditional(model1, which))

    k1, k2 = jr.split(key)
    d1, d2 = jr.normal(k1, f1.shape), jr.normal(k2, f1.shape)
    d1 /= jnp.linalg.norm(d1)
    d2 -= jnp.dot(d2, d1) * d1
    d2 /= jnp.linalg.norm(d2)

    if normalise:
        d1 = d1 * (jnp.linalg.norm(f1) / (jnp.linalg.norm(d1) + 1e-10))
        d2 = d2 * (jnp.linalg.norm(f1) / (jnp.linalg.norm(d2) + 1e-10))

    xvals = jnp.linspace(-2, 2, num_points)
    yvals = jnp.linspace(-2, 2, num_points)
    xmesh, ymesh = jnp.meshgrid(xvals, yvals)

    losses = evaluate_loss_on_mesh_grid(
        xmesh,
        ymesh,
        f1,
        meta,
        model1,
        loss_fn,
        conf,
        data,
        t_save,
        masks,
        last_valid_index,
        d1,
        d2,
        which
    )

    if model2 is not None:
        f2, _ = flatten_tensor_list(get_params_conditional(model2, which))
        proj_x, proj_y = project_2d(f2 - f1, d1, d2)
    else:
        proj_x, proj_y = None, None
    return losses, xmesh, ymesh, proj_x, proj_y



def evaluate_loss_on_hessian_grid(
    model,
    dir1,
    dir2,
    loss_fn,
    conf,
    data,
    t_save,
    masks,
    last_valid_index,
    num_points=20,
    which: str = "full",
):
    flat_params, metadata = flatten_tensor_list(get_params_conditional(model, which))
    xmesh, ymesh = jnp.meshgrid(
        jnp.linspace(-1, 1, num_points),
        jnp.linspace(-1, 1, num_points)
    )
    losses = evaluate_loss_on_mesh_grid(
        xmesh,
        ymesh,
        flat_params,
        metadata,
        model,
        loss_fn,
        conf,
        data,
        t_save,
        masks,
        last_valid_index,
        dir1,
        dir2,
        which
    )
    return losses, xmesh, ymesh


def plot_loss_landscape(
    xmesh,
    ymesh,
    losses,
    mode: str,
    levels: int = 30,
    xcoords: list=None,
    ycoords: list=None,
    proj_model_x=None,
    proj_model_y=None,
    title="Loss Landscape",
    color_labels: list=None,
    cmap: str ='tab10',
):
    fig, ax = plt.subplots(figsize=(8, 5))
    contour = ax.contourf(
        xmesh,
        ymesh,
        np.log(losses.reshape(xmesh.shape)),
        levels=levels,
        cmap='viridis'
    )

    if mode == "random":
        if proj_model_x is not None and proj_model_y is not None:
            ax.plot(0, 0, 'r*', markersize=10, label="Model 1 (center)")
            ax.plot(proj_model_x, proj_model_y, 'bo-', markersize=5, label='Model 2')
        xlabel = "Random Direction 1"
        ylabel = "Random Direction 2"
    else:
        if xcoords is not None and ycoords is not None:
            if color_labels is not None:
                unique_labels = sorted(set(color_labels))
                color_map = cm.get_cmap(cmap, len(unique_labels))
                label_to_color = {label: color_map(i) for i, label in enumerate(unique_labels)}

                df = pd.DataFrame({"x": xcoords, "y": ycoords, "cl_stage": color_labels})
                for label in unique_labels:
                    group = df[df["cl_stage"] == label]
                    ax.plot(group["x"], group["y"], '-', color=label_to_color[label], label=f"CL Stage {label}")
                    ax.plot(group["x"], group["y"], 'o', color=label_to_color[label], markersize=4)
            else:
                ax.plot(xcoords, ycoords, 'bo-', markersize=5)
                # Override start marker with red square
                ax.plot(xcoords[0], ycoords[0], 'rs-', markersize=10, label='Start')
        if mode == "PCA":
            xlabel = "PC1"
            ylabel = "PC2"
        elif mode == "hessian":
            xlabel = "Hessian Direction 1"
            ylabel = "Hessian Direction 2"
            ax.plot(0, 0, color="gold", marker='*', markersize=10, label="End")
        else:
            raise ValueError(f"Unknown mode: {mode}")

    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("Log Loss")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if mode == "PCA":
        if xcoords is not None:
            x_margin = 0.1 * (max(xcoords) - min(xcoords))
            ax.set_xlim(min(xcoords) - x_margin, max(xcoords) + x_margin)
        if ycoords is not None:
            y_margin = 0.1 * (max(ycoords) - min(ycoords))
            ax.set_ylim(min(ycoords) - y_margin, max(ycoords) + y_margin)

    ax.legend(bbox_to_anchor=(1.30, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def visualize_hessian_eigvec_landscape(
    model_dict,
    conf,
    t_save_windows,
    masks,
    data,
    curriculum_stage_dict,
    base_loss_fn,
    loss_fn,
    hessian_fn,
    which="full",
    num_points=20,
    plot_pairs: bool = True
):
    # Initialise variables
    xmesh_hess, ymesh_hess, losses_hess, final_model = None, None, None, None
    last_valid_index = 1
    dir1, dir2 = None, None
    active_hessian_idx = None

    for cl_stage in model_dict.keys():
        # Sort by training epoch and get final model
        sorted_epochs = sorted(model_dict[cl_stage])
        final_epoch = sorted_epochs[-1]
        final_model = model_dict[cl_stage][final_epoch]

        # Get last valid index, save windows and masks for the current curriculum stage (if any)
        if conf.curriculum_learning:
            last_valid_index = curriculum_stage_dict[cl_stage]["last_valid_index"]
            t_save_windows = curriculum_stage_dict[cl_stage]["time"]
            masks = curriculum_stage_dict[cl_stage]["mask"]
            active_hessian_idx = update_hessian_index_mask_multiple_curriculum(
                final_model,
                eqx.is_array,
                last_valid_index
            )

        # === Hessian Landscape ===
        dir1, dir2 = compute_hessian_directions(
            final_model,
            conf,
            t_save_windows,
            masks,
            data,
            last_valid_index=last_valid_index,
            base_loss_fn=base_loss_fn,
            loss_hessian_fn=hessian_fn,
            active_idx=active_hessian_idx,
        )
        losses_hess, xmesh_hess, ymesh_hess = evaluate_loss_on_hessian_grid(
            final_model,
            dir1,
            dir2,
            loss_fn,
            conf,
            data,
            t_save_windows,
            masks,
            last_valid_index,
            which=which,
            num_points=num_points
        )

        plot_loss_landscape(
            xmesh_hess,
            ymesh_hess,
            losses_hess,
            mode="hessian",
            title=f"Loss Landscape (Hessian Directions) - CL Stage {cl_stage}"
        )

        if plot_pairs:
            # === Project all checkpoints onto final Hessian landscape ===
            final_params, _ = flatten_tensor_list(get_params_conditional(final_model, "full"))
            coord_dfs = []
            for epoch in sorted_epochs:
                model_i = model_dict[cl_stage][epoch]
                params_i = get_params_conditional(model_i, "full")
                flat_i, _ = flatten_tensor_list(params_i)
                dx, dy = project_2d(flat_i - final_params, dir1, dir2)
                coord_dfs.append(pd.DataFrame({"x": [dx], "y": [dy], "stage": [epoch]}))

            df_proj = pd.concat(coord_dfs, ignore_index=True)
            df_proj.sort_values(by="stage", inplace=True)
            plot_loss_landscape(
                xmesh_hess,
                ymesh_hess,
                losses_hess,
                mode="hessian",
                xcoords=df_proj["x"].tolist(),
                ycoords=df_proj["y"].tolist(),
                title=f"Loss Landscape (Model Checkpoints) - CL Stage {cl_stage}"
            )

    # Sort by training epoch and get final model
    model_dict_flattened = {}
    stage_labels = []
    for cl_stage in model_dict:
        for epoch in model_dict[cl_stage]:
            model_dict_flattened[(cl_stage, epoch)] = model_dict[cl_stage][epoch]
            stage_labels.append(cl_stage)

    sorted_keys = sorted(model_dict_flattened, key=lambda x: (x[0], x[1]))  # sort by cl_stage then epoch
    flat_models = {k: model_dict_flattened[k] for k in sorted_keys}

    df_coords, (xcoords, ycoords), (pc1, pc2) = project_trajectory(flat_models, which=which)
    df_coords["cl_stage"] = [key[0] for key in sorted_keys[:-1]]  # align stage labels, skipping the last model

    losses_pca = evaluate_loss_on_pca_grid(
        final_model,
        pc1,
        pc2,
        xcoords,
        ycoords,
        loss_fn,
        conf,
        data,
        t_save_windows,
        masks,
        last_valid_index,
        which=which,
        num_points=num_points
    )
    xmesh_pca, ymesh_pca = np.meshgrid(
        np.linspace(min(xcoords), max(xcoords), num_points),
        np.linspace(min(ycoords), max(ycoords), num_points)
    )
    plot_loss_landscape(
        xmesh_pca,
        ymesh_pca,
        losses_pca,
        mode="PCA",
        xcoords=df_coords["x"].tolist(),
        ycoords=df_coords["y"].tolist(),
        color_labels=df_coords["cl_stage"].tolist(),
        title="Loss Landscape (PCA Directions by CL Stage)"
    )

    # === Project all checkpoints onto final Hessian landscape ===
    coord_dfs = []
    for cl_stage in model_dict.keys():
        final_model = model_dict[cl_stage][sorted(model_dict[cl_stage])[-1]]
        final_params, _ = flatten_tensor_list(get_params_conditional(final_model, "full"))

        for epoch, model in model_dict[cl_stage].items():
            params = get_params_conditional(model, "full")
            flat_params, _ = flatten_tensor_list(params)
            dx, dy = project_2d(flat_params - final_params, dir1, dir2)
            coord_dfs.append(
                pd.DataFrame({
                    "x": [dx],
                    "y": [dy],
                    "stage": [epoch],
                    "cl_stage": [cl_stage]
                })
            )

    df_proj = pd.concat(coord_dfs, ignore_index=True)
    plot_loss_landscape(
        xmesh_hess,
        ymesh_hess,
        losses_hess,
        mode="hessian",
        xcoords=df_proj["x"].tolist(),
        ycoords=df_proj["y"].tolist(),
        color_labels=df_proj["cl_stage"].tolist(),
        title="Loss Landscape (Model Checkpoints by CL Stage)"
    )


def test_curriculum_stage_convexity(
    model_dict,
    conf,
    curriculum_stage_dict,
    data,
    t_save_windows,
    masks,
    loss_fn: Callable,
    which: str = "full",
    num_dirs: int = 1000,
    mode: str = "local",
    sampling: Optional[str] = "lh",
    filter_normalise: Optional[bool] = True
):
    # Initialise variables
    theta1_list, theta2_list, alpha_list, directions, alphas = [], [], [], [], []
    flat_ref = None

    # Get shape of flattened params from any model
    example_stage = list(model_dict.keys())[-1]
    example_epoch = sorted(model_dict[example_stage])[-1]
    example_model = model_dict[example_stage][example_epoch]
    flat_example_params, metadata = flatten_tensor_list(get_params_conditional(example_model, which))
    param_shape = flat_example_params.shape

    # Seed PRNG from multistart_job and get master key
    key = jr.PRNGKey(conf.multistart_job)

    if mode == "global":
        key_theta1, key_dirs, key_alpha = jr.split(key, 3)
        # For global convexity, generate random θ₁ and θ₂ (defined as θ₁ + d where d is unit vector)
        if sampling == "lh":
            # Sample using Latin Hypercube [0, 1) and rescale to [-1, 1)
            theta1_list = 2 * qmc.LatinHypercube(
                d=param_shape[0],
                seed=int(key_theta1[0]) << 32 | int(key_theta1[1])  # turn JAX key into int seed
            ).random(num_dirs) - 1
        else:
            # Sample using normal distribution
            theta1_list = [jr.normal(k, param_shape) for k in jr.split(key_theta1, num_dirs)]
        if filter_normalise:
            # Filter-normalise θ₁ to have the same norm as the example model -> model manifold
            norm_target = jnp.linalg.norm(flat_example_params)
            theta1_list = [(v / jnp.linalg.norm(v)) * norm_target for v in theta1_list]
        dir_list = [jr.normal(k, param_shape) for k in jr.split(key_dirs, num_dirs)]
        dir_list = [d / jnp.linalg.norm(d) for d in dir_list]
        theta2_list = [theta1 + d for theta1, d in zip(theta1_list, dir_list)]
        # Precompute corresponding alphas ∈ (0, 1)
        alpha_list = jr.uniform(key_alpha, shape=(num_dirs,), minval=0.01, maxval=0.99)
    else:
        key_dirs, key_alphas = jr.split(key)
        # For local convexity, generate random perturbations to theta_1=final model
        direction_keys = jr.split(key_dirs, num_dirs)
        directions = [jr.normal(k, param_shape) for k in direction_keys]
        directions = [d / jnp.linalg.norm(d) for d in directions]
        alphas = jr.uniform(key_alphas, shape=(num_dirs,), minval=0.01, maxval=0.99)

    results = {}

    for cl_stage in model_dict:
        sorted_epochs = sorted(model_dict[cl_stage])
        final_model = model_dict[cl_stage][sorted_epochs[-1]]

        if conf.curriculum_learning:
            last_valid_index = curriculum_stage_dict[cl_stage]["last_valid_index"]
            t_save = curriculum_stage_dict[cl_stage]["time"]
            stage_masks = curriculum_stage_dict[cl_stage]["mask"]
        else:
            last_valid_index = 1
            t_save = t_save_windows
            stage_masks = masks

        if mode == "local":
            flat_ref, metadata = flatten_tensor_list(get_params_conditional(final_model, which))

        count_convex = 0
        pbar = tqdm(range(num_dirs), desc=f"CL Stage {cl_stage}", unit="dir")
        for i in pbar:
            if mode == "global":
                # Random θ₁ and θ₂
                flat_ref = theta1_list[i]
                flat_shift = theta2_list[i]
                alpha = alpha_list[i]
            else:
                direction = directions[i]
                alpha = alphas[i]
                # θ₂ = θ₁ + d
                flat_shift = flat_ref + direction

            # Get θ₂ model
            shifted_params = unflatten_tensor_list(flat_shift, metadata)
            shifted_model = update_model_from_params(final_model, shifted_params, which)

            # θ_mix = αθ₁ + (1 - α)θ₂
            flat_mix = alpha * flat_ref + (1 - alpha) * flat_shift
            mixed_params = unflatten_tensor_list(flat_mix, metadata)
            mixed_model = update_model_from_params(final_model, mixed_params, which)

            def get_loss(model):
                params, static = eqx.partition(model, eqx.is_array)
                loss, _ = loss_fn(
                    params,
                    static,
                    t_save,
                    stage_masks,
                    data,
                    last_valid_index,
                    conf.mlp_reg_exp,
                    conf.mlp_reg_strength,
                    conf.output_reg,
                    conf.multiple_shooting_penalty,
                    conf.shooting_mode
                )
                return loss

            # Evaluate losses
            loss_ref = get_loss(
                final_model if mode == "local"
                else update_model_from_params(
                    final_model, unflatten_tensor_list(flat_ref, metadata), which
                )
            )
            loss_shift = get_loss(shifted_model)
            loss_mix = get_loss(mixed_model)

            # Convexity condition check
            if loss_mix <= alpha * loss_ref + (1 - alpha) * loss_shift:
                count_convex += 1

            pbar.set_postfix_str(f"Convex {count_convex}/{i+1}")

        results[cl_stage] = count_convex / num_dirs

    return results


def plot_convexity_fractions(
    results_dict,
    title="Convexity Check per Curriculum Stage",
    mode: str = "local"
):
    cl_stages = list(results_dict.keys())
    fractions = list(results_dict.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(cl_stages, fractions, color='skyblue', edgecolor='black')

    # Annotate each bar with its value
    for bar, fraction in zip(bars, fractions):
        height = bar.get_height()
        ax.annotate(f"{fraction:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of Convex Directions")
    ax.set_xlabel("Curriculum Stage")
    ax.set_title(title + f" - {mode}")
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_lotka_volterra_dynamic_param_trajectory(
    model_dict: dict,
    ground_truth_dynamic_params: Array,
    rescaling_args: dict,
    title: str = "LV dynamic parameter trajectory",
):
    # Initialise variables to track dynamic parameters across curriculum stages and epochs
    records = []
    step = 0

    # Order: sort by curriculum stage first, then epoch
    for cl_stage in sorted(model_dict):
        epoch_dict = model_dict[cl_stage]
        for epoch in sorted(epoch_dict):
            model = epoch_dict[epoch]
            # Extract dynamic parameters, process them to linear scale and rescale if using timepoint normalisation
            dyn = process_mechanistic_params(
                model.func.dynamic_parameters, model.func.options
            ) / rescaling_args["time_rescaling"]
            records.append(
                {
                    "step": step,
                    "cl_stage": cl_stage,
                    "epoch": epoch,
                    "alpha": dyn[0],
                    "delta": dyn[1],
                }
            )
            step += 1

    # Create DataFrame
    df = pd.DataFrame.from_records(records)

    # Plot: one subplot per dynamic parameter, coloured by curriculum stage
    cl_stages = sorted(df["cl_stage"].unique())
    cmap = cm.get_cmap("tab10", len(cl_stages))
    stage_to_color = {stage: cmap(i) for i, stage in enumerate(cl_stages)}

    fig, axes = plt.subplots(
        2, 1, figsize=(8, 5), sharex=True
    )

    for i, pname in enumerate(["alpha", "delta"]):
        ax = axes[i]

        # Plot per curriculum stage
        for stage in cl_stages:
            ax.plot(
                df[df["cl_stage"] == stage]["step"],
                df[df["cl_stage"] == stage][pname],
                "-o",
                markersize=3,
                label=f"CL stage {stage}" if i == 0 else None,
                color=stage_to_color[stage],
                alpha=0.8,
            )

        # Ground-truth horizontal line
        ax.axhline(
            float(ground_truth_dynamic_params[i]),
            linestyle="--",
            linewidth=1.5,
            color="black",
            label="Ground truth" if i == 0 else None,
        )

        # Mark starting and final points
        ax.plot(
            df.iloc[0]["step"],
            df.iloc[0][pname],
            marker="s",
            markersize=6,
            color="red",
            label="Initial value" if i == 0 else None,
        )
        ax.plot(
            df.iloc[-1]["step"],
            df.iloc[-1][pname],
            marker="*",
            markersize=8,
            color="gold",
            label="Final value" if i == 0 else None,
        )

        ax.set_ylabel(f"{pname}")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Training step (curriculum stage & epoch order)")
    fig.suptitle(title, fontsize=14)

    # Build shared legend from the first axis
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(len(labels), 4),
        )

    plt.tight_layout()
    plt.show()


def evaluate_lv_loss_on_param_grid(
    base_model: UDEProblem,
    loss_fn: Callable,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    conf,
    data: np.ndarray,
    t_save,
    masks,
    last_valid_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Helper to evaluate the loss on a 2D grid in Lotka–Volterra dynamic parameter space (alpha, delta).

    Parameters
    ----------
    base_model :
        Model whose non-dynamic parameters (MLP weights, initial values of shooting windows) are held fixed.
        Only dynamic parameters (base_model.func.dynamic_parameters) are varied over the grid.
    loss_fn, conf, data, t_save, masks, last_valid_index:
        Loss function callable and arguments necessary for loss computation.
    x_vals, y_vals :
        1D arrays of values to use for alpha (x) and delta (y), used to build loss surface mesh.

    Returns
    -------
    xmesh, ymesh, losses :
        2D meshes for alpha/delta and the corresponding loss values.
    """

    xmesh, ymesh = np.meshgrid(x_vals, y_vals)
    losses = np.zeros_like(xmesh, dtype=float)

    # Build loss surface
    for i in range(xmesh.shape[0]):
        for j in range(xmesh.shape[1]):
            # Replace dynamic parameters with current grid point
            new_model = eqx.tree_at(
                lambda m: m.func.dynamic_parameters,
                base_model,
                jnp.array([xmesh[i, j], ymesh[i, j]]),
            )
            params, static = eqx.partition(new_model, eqx.is_array)
            loss, _ = loss_fn(
                params,
                static,
                t_save,
                masks,
                data,
                last_valid_index,
                conf.mlp_reg_exp,
                conf.mlp_reg_strength,
                conf.output_reg,
                conf.multiple_shooting_penalty,
                conf.shooting_mode,
            )
            losses[i, j] = loss

    return xmesh, ymesh, losses


def _compute_stage_loss_surface(
    base_model: UDEProblem,
    stage_key,
    conf,
    curriculum_stage_dict,
    data: np.ndarray,
    t_save_windows,
    masks,
    loss_fn: Callable,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
):
    """Shared helper to get (xmesh, ymesh, losses) for a given curriculum stage."""
    if conf.curriculum_learning:
        last_valid_index = curriculum_stage_dict[stage_key]["last_valid_index"]
        t_save = curriculum_stage_dict[stage_key]["time"]
        stage_masks = curriculum_stage_dict[stage_key]["mask"]
    else:
        last_valid_index = 1
        t_save = t_save_windows
        stage_masks = masks

    xmesh, ymesh, losses = evaluate_lv_loss_on_param_grid(
        base_model=base_model,
        loss_fn=loss_fn,
        conf=conf,
        data=data,
        t_save=t_save,
        masks=stage_masks,
        last_valid_index=last_valid_index,
        x_vals=x_vals,
        y_vals=y_vals,
    )
    return xmesh, ymesh, losses


def visualize_lv_loss_around_true_dyn_params(
    model_dict: dict,
    conf,
    curriculum_stage_dict,
    data: np.ndarray,
    t_save_windows,
    masks,
    loss_fn: Callable,
    ground_truth_dynamic_params: Array,
    rescaling_args: dict,
    num_points: int = 21,
    plot_in_log: bool = False,
    title: str = "LV Loss Landscape around true dynamic parameters",
):
    """
    Builds a single 2D loss landscape along Lotka-Volterra parameters (alpha, delta)
    centred at the provided ground truth parameters. Overlays on such loss landscape
    the training trajectory (i.e. model dynamic parameters) across curriculum stages.

    The grid is chosen so that it is large enough to include all observed
    (alpha, delta) values across all stages/epochs (in particular the sampled start).
    Parameter grid can be plotted in log10 space (where they are typically estimated) using the plot_in_log flag.

    Assumes:
      - conf.dynamical_model == "lotka_volterra"
      - ground_truth_dynamic_params in linear scale, same order as model.func.dynamic_parameters (alpha, delta).
    """
    gt_alpha, gt_delta = ground_truth_dynamic_params

    # Collect all (alpha, delta) positions across all curriculum stages and epochs
    all_alphas, all_deltas = [], []
    traj_by_stage = {}

    for cl_stage in sorted(model_dict):
        epoch_dict = model_dict[cl_stage]
        sorted_epochs = sorted(epoch_dict)

        alphas_stage, deltas_stage = [], []
        for epoch in sorted_epochs:
            model = epoch_dict[epoch]
            a, d = process_mechanistic_params(
                model.func.dynamic_parameters, model.func.options
            ) / rescaling_args["time_rescaling"]
            alphas_stage.append(a)
            deltas_stage.append(d)
            all_alphas.append(a)
            all_deltas.append(d)

        traj_by_stage[cl_stage] = {
            "epochs": sorted_epochs,
            "alphas": alphas_stage,
            "deltas": deltas_stage,
        }

    # Convert to log10 scale if specified
    if plot_in_log:
        gt_alpha = np.log10(gt_alpha)
        gt_delta = np.log10(gt_delta)
        all_alphas = [np.log10(a) for a in all_alphas]
        all_deltas = [np.log10(d) for d in all_deltas]

    # Define grid centred at ground truth, big enough to contain the entire trajectory
    # Augment ranges with 20% margin and build grid
    x_half_range = 1.2 * max(abs(a - gt_alpha) for a in all_alphas)
    y_half_range = 1.2 * max(abs(d - gt_delta) for d in all_deltas)
    x_vals = np.linspace(gt_alpha - x_half_range, gt_alpha + x_half_range, num_points)
    y_vals = np.linspace(gt_delta - y_half_range, gt_delta + y_half_range, num_points)

    # Use the final model of the last curriculum stage as base_model for the loss surface
    last_stage = sorted(model_dict)[-1]
    last_epoch = sorted(model_dict[last_stage])[-1]
    base_model = model_dict[last_stage][last_epoch]

    # Compute single loss surface
    xmesh, ymesh, losses = _compute_stage_loss_surface(
        base_model=base_model,
        stage_key=last_stage,
        conf=conf,
        curriculum_stage_dict=curriculum_stage_dict,
        data=data,
        t_save_windows=t_save_windows,
        masks=masks,
        loss_fn=loss_fn,
        x_vals=x_vals,
        y_vals=y_vals,
    )

    # Plot with trajectory overlaid
    fig, ax = plt.subplots(figsize=(7, 6))
    contour = ax.contourf(
        xmesh,
        ymesh,
        np.log(losses),
        levels=40,
        cmap="viridis",
        # norm=LogNorm()
    )
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("Log loss")

    # Colour-code trajectory by curriculum stage
    stages = sorted(traj_by_stage.keys())
    cmap = cm.get_cmap("tab10", len(stages))
    stage_to_color = {stage: cmap(i) for i, stage in enumerate(stages)}

    # Track global start & final for highlighting
    first_stage = stages[0]
    first_a = traj_by_stage[first_stage]["alphas"][0]
    first_d = traj_by_stage[first_stage]["deltas"][0]

    last_stage = stages[-1]
    last_a = traj_by_stage[last_stage]["alphas"][-1]
    last_d = traj_by_stage[last_stage]["deltas"][-1]

    if plot_in_log:
        first_a = np.log10(first_a)
        first_d = np.log10(first_d)
        last_a = np.log10(last_a)
        last_d = np.log10(last_d)

    for stage in stages:
        if plot_in_log:
            alphas = [np.log10(a) for a in traj_by_stage[stage]["alphas"]]
            deltas = [np.log10(d) for d in traj_by_stage[stage]["deltas"]]
        else:
            alphas = traj_by_stage[stage]["alphas"]
            deltas = traj_by_stage[stage]["deltas"]
        ax.plot(
            alphas,
            deltas,
            "-o",
            markersize=3,
            color=stage_to_color[stage],
            label=f"CL stage {stage}",
        )

    # Ground truth, global start and global final markers
    ax.plot(gt_alpha, gt_delta, "x", color="black", markersize=8, label="Ground truth")
    ax.plot(first_a, first_d, "s", color="red", markersize=7, label="Start")
    ax.plot(last_a, last_d, "*", color="gold", markersize=9, label="Final")

    ax.set_xlabel("alpha")
    ax.set_ylabel("delta")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_lv_loss_on_dyn_params_per_stage(
    model_dict,
    conf,
    curriculum_stage_dict,
    data,
    t_save_windows,
    masks,
    loss_fn: Callable,
    ground_truth_dynamic_params: Array,
    rescaling_args: dict,
    num_points: int = 21,
    plot_in_log: bool = False,
    title_prefix: str = "LV Loss Landscape (stage-centred)",
):
    """
    For each curriculum stage:
      - Takes the final model of that stage as base model;
      - Builds a 2D grid in (alpha, delta) centred around the final dynamic
        parameters for that curriculum stage, with a range wide enough to include also
        the dynamic parameters for the curriculum stage start and middle epochs;
      - Plots the loss contour and marks:
        - start, middle, final dynamic parameters for that stage,
        - the provided ground-truth parameters.
    Parameter grid can be plotted in log10 space (where they are typically estimated) using the plot_in_log flag.

    Assumes:
      - conf.dynamical_model == "lotka_volterra"
      - ground_truth_dynamic_params in linear scale, same order as model.func.dynamic_parameters (alpha, delta).
    """

    gt_alpha, gt_delta = ground_truth_dynamic_params

    if plot_in_log:
        gt_alpha = np.log10(gt_alpha)
        gt_delta = np.log10(gt_delta)

    stage_points = {}
    all_x = [gt_alpha]
    all_y = [gt_delta]

    for cl_stage in sorted(model_dict):
        epoch_dict = model_dict[cl_stage]
        sorted_epochs = sorted(epoch_dict)
        if len(sorted_epochs) < 1:
            continue

        start_epoch = sorted_epochs[0]
        mid_epoch = sorted_epochs[len(sorted_epochs) // 2]
        final_epoch = sorted_epochs[-1]

        start_model = epoch_dict[start_epoch]
        mid_model = epoch_dict[mid_epoch]
        final_model = epoch_dict[final_epoch]

        a_start, d_start = process_mechanistic_params(
            start_model.func.dynamic_parameters, start_model.func.options
        ) / rescaling_args["time_rescaling"]
        a_mid, d_mid = process_mechanistic_params(
            mid_model.func.dynamic_parameters, mid_model.func.options
        ) / rescaling_args["time_rescaling"]
        a_final, d_final = process_mechanistic_params(
            final_model.func.dynamic_parameters, final_model.func.options
        ) / rescaling_args["time_rescaling"]

        # If needed, convert to log10 scale
        if plot_in_log:
            a_start = np.log10(a_start)
            d_start = np.log10(d_start)
            a_mid = np.log10(a_mid)
            d_mid = np.log10(d_mid)
            a_final = np.log10(a_final)
            d_final = np.log10(d_final)

        stage_points[cl_stage] = {
            "a_start": a_start,
            "d_start": d_start,
            "a_mid": a_mid,
            "d_mid": d_mid,
            "a_final": a_final,
            "d_final": d_final,
            "final_epoch": final_epoch,
        }

        all_x.extend([a_start, a_mid, a_final])
        all_y.extend([d_start, d_mid, d_final])

    # Build global grid bounds (same for every stage)
    x_min_raw, x_max_raw = min(all_x), max(all_x)
    y_min_raw, y_max_raw = min(all_y), max(all_y)

    # Add 20% margin to min-max ranges
    x_min = x_min_raw - 0.2 * (x_max_raw - x_min_raw)
    x_max = x_max_raw + 0.2 * (x_max_raw - x_min_raw)
    y_min = y_min_raw - 0.2 * (y_max_raw - y_min_raw)
    y_max = y_max_raw + 0.2 * (y_max_raw - y_min_raw)

    x_vals = np.linspace(x_min, x_max, num_points)
    y_vals = np.linspace(y_min, y_max, num_points)

    # Loop through curriculum stages and plot
    for cl_stage in sorted(model_dict):
        if cl_stage not in stage_points:
            continue

        epoch_dict = model_dict[cl_stage]
        final_epoch = stage_points[cl_stage]["final_epoch"]
        final_model = epoch_dict[final_epoch]

        # Compute loss surface for this curriculum stage on the global grid
        xmesh, ymesh, losses = _compute_stage_loss_surface(
            base_model=final_model,
            stage_key=cl_stage,
            conf=conf,
            curriculum_stage_dict=curriculum_stage_dict,
            data=data,
            t_save_windows=t_save_windows,
            masks=masks,
            loss_fn=loss_fn,
            x_vals=x_vals,
            y_vals=y_vals,
        )

        sp = stage_points[cl_stage]
        a_start, d_start = sp["a_start"], sp["d_start"]
        a_mid, d_mid = sp["a_mid"], sp["d_mid"]
        a_final, d_final = sp["a_final"], sp["d_final"]

        # ---- 5) Plot ----
        fig, ax = plt.subplots(figsize=(6, 5))
        contour = ax.contourf(
            xmesh,
            ymesh,
            np.log(losses),
            levels=30,
            cmap="viridis",
        )
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label("Log loss")

        # Mark start, mid, final, ground truth
        ax.plot(a_start, d_start, "s", color="red", markersize=6, label="Start")
        ax.plot(a_mid, d_mid, "o", color="orange", markersize=5, label="Mid")
        ax.plot(a_final, d_final, "*", color="gold", markersize=8, label="Final")
        ax.plot(gt_alpha, gt_delta, "x", color="black", markersize=7, label="Ground truth")

        ax.set_xlabel("alpha" if not plot_in_log else "log10(alpha)")
        ax.set_ylabel("delta" if not plot_in_log else "log10(delta)")
        ax.set_title(f"{title_prefix} – CL stage {cl_stage}")
        ax.legend(loc="best")
        ax.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.show()
