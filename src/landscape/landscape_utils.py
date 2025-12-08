import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jaxtyping import Array
from scipy.stats import qmc
from sklearn.decomposition import PCA
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
    fig, ax = plt.subplots(figsize=(6, 5))
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
                ax.plot(xcoords, ycoords, 'bo-', markersize=5, label='Projected Models')
        if mode == "PCA":
            xlabel = "PC1"
            ylabel = "PC2"
        elif mode == "hessian":
            xlabel = "Hessian Direction 1"
            ylabel = "Hessian Direction 2"
            ax.plot(0, 0, 'r*', markersize=10, label="Final Model")
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

    ax.legend()
    plt.tight_layout()
    plt.show()


def visualize_landscape(
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
            title=f"Loss Landscape (Hessian Directions) - CL Stage {cl_stage}",
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
            plot_loss_landscape(
                xmesh_hess,
                ymesh_hess,
                losses_hess,
                mode="hessian",
                xcoords=df_proj["x"].tolist(),
                ycoords=df_proj["y"].tolist(),
                title=f"Loss Landscape (Model Checkpoints) - CL Stage {cl_stage}"
            )

    # last_valid_index = 1
    # final_model = model_dict[max(model_dict)][sorted(model_dict[max(model_dict)])[-1]]

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
