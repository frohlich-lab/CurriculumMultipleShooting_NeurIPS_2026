import diffrax
import equinox as eqx
import git
import itertools as itt
import jax.numpy as jnp
import jax
import numpy as np
import optax
import wandb
from diffrax import SaveAt

from functools import partial
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree
from src.common import Conf, dynamic_param_labels, hessian_metrics_wandb
from src.model_training.model_util import set_weight_bias
from src.ude_problem.ude_problem import UDEProblem
from typing import Callable, Dict, List, Optional, Tuple, Union


def augment_u0(u0):
    """
    Augment the initial condition(s) with additional zero(es) element to compute
    network contribution regularisation.
    """
    # Dynamically compute the shape for the new zeros array
    zeros_shape = u0.shape[:-1] + (1,)
    # Concatenate along the last axis
    return jnp.concatenate([u0, jnp.zeros(zeros_shape)], axis=-1)


# Utilities to prepare for training
def get_filter_spec(model: UDEProblem, conf: Conf, optimiser: str = "adam"):
    # Depending on conf settings, freeze mechanistic params/neural network or keep everything learnable
    if (
            ((optimiser == "adam") and conf.freeze_mechanistic_params_adam) or
            ((optimiser == "bfgs") and conf.freeze_mechanistic_params_optx)
    ):
        # Create filter_spec allowing for training of mechanistic params only or neural network only
        filter_spec_freeze_mechanistic = jax.tree.map(lambda x: eqx.is_array(x), model)
        # filter_spec which freezes dynamic (mechanistic) parameters
        filter_spec_freeze_mechanistic = eqx.tree_at(
            lambda x: x.func.dynamic_parameters,
            filter_spec_freeze_mechanistic,
            replace=jax.tree.map(lambda _: False, filter_spec_freeze_mechanistic.func.dynamic_parameters)
        )
        return filter_spec_freeze_mechanistic
    elif (
            ((optimiser == "adam") and conf.freeze_nn_adam) or
            ((optimiser == "bfgs") and conf.freeze_nn_optx)
    ):
        # filter_spec which freezes the neural network
        filter_spec_freeze_nn = jax.tree.map(lambda x: False, model)
        filter_spec_freeze_nn = eqx.tree_at(
            lambda x: x.func.dynamic_parameters,
            filter_spec_freeze_nn,
            replace=jax.tree.map(lambda _: True, filter_spec_freeze_nn.func.dynamic_parameters)
        )
        return filter_spec_freeze_nn
    else:
        return eqx.is_array


# Scheduler from Fabian's NeuralCoarseGraining
def get_scheduler(
        conf: Conf,
        n_epochs: int,
):
    """Get the learning rate scheduler.

    Parameters
    ----------
    conf : configuration object
    n_epochs : int - total number of training epochs

    Returns
    ----------
    optax.sgdr_schedule
        The learning rate scheduler.
    """
    if (conf.opt_steps > 0) and (conf.opt_mult > 1):
        if conf.opt_sched_type != "cos":
            raise ValueError(
                f"Multiple cosine annealing schedules not supported with schedule type: {conf.opt_sched_type}!"
            )
        epochs_per_schedule = np.array([
            conf.opt_steps * (conf.opt_mult ** i)
            for i in range(int(n_epochs // conf.opt_steps))
            if conf.opt_steps * (conf.opt_mult ** i) <= n_epochs
        ])
        schedules = [
            {
                'init_value': conf.max_lrate / conf.lrate_span * conf.lrate_decay ** i_schedule,
                'peak_value': conf.max_lrate * conf.lrate_decay ** i_schedule,
                'warmup_steps': int(
                    (conf.opt_steps * (conf.opt_mult ** i_schedule))
                    * conf.warmup_fct
                ),
                'decay_steps': int(conf.opt_steps * (conf.opt_mult ** i_schedule)),
                'end_value': conf.max_lrate / conf.lrate_span * conf.lrate_decay ** (i_schedule + 1),
            }
            for i_schedule in range(len(epochs_per_schedule))
        ]
        return optax.sgdr_schedule(schedules)
    elif (conf.opt_steps == 0) and (conf.opt_mult == 0):
        if conf.opt_sched_type == "exp":
            # Single schedule with warmup and exponential decay
            schedule = {
                "init_value": conf.max_lrate / conf.lrate_span,  # start of annealing
                "peak_value": conf.max_lrate,  # end of annealing
                "warmup_steps": int(n_epochs * conf.warmup_fct),
                "transition_begin": 0,
                "transition_steps": 1,  # rather than decay_rate ** n_epochs, decay_rate ** ((count - transition_begin) / transition_steps)
                "decay_rate": conf.lrate_decay,
                "end_value": conf.max_lrate * conf.lrate_decay ** n_epochs,  # lower bound when to stop
            }
            return optax.schedules.warmup_exponential_decay_schedule(**schedule)
        elif conf.opt_sched_type == "lin":  # Single schedule with linear warmup and decay
            if conf.lrate_decay == 1 and conf.lrate_span == 1 and conf.warmup_fct == 0:
                return optax.constant_schedule(conf.max_lrate)
            # In this case, use conf.lrate_decay as a divider of the initial value to determine the end learning rate
            if conf.lrate_decay < 1:
                decay_factor = 1e2  # default to decaying to 1/100 of the starting value (pre-warmup)
            else:
                decay_factor = conf.lrate_decay

            init_value = conf.max_lrate / conf.lrate_span
            final_value = init_value / decay_factor

            linear_schedule = {
                "transition_steps": n_epochs,
                "peak_value": conf.max_lrate,
                "pct_start": conf.warmup_fct,
                "pct_final": conf.warmup_fct + ((init_value - conf.max_lrate) / (final_value - conf.max_lrate)) * (
                            1.0 - conf.warmup_fct),
                "div_factor": conf.lrate_span,
                "final_div_factor": decay_factor,
            }
            return optax.schedules.linear_onecycle_schedule(**linear_schedule)
        else:  # default: cosine annealing
            schedule = {
                "init_value": conf.max_lrate / conf.lrate_span,
                "peak_value": conf.max_lrate,
                "warmup_steps": int(n_epochs * conf.warmup_fct),
                "decay_steps": n_epochs,
                "end_value": conf.max_lrate * conf.lrate_decay ** n_epochs,
            }
            return optax.sgdr_schedule([schedule])
    else:
        raise ValueError(
            f"Invalid configuration for LR schedule: opt_steps={conf.opt_steps}, opt_mult={conf.opt_mult}!"
        )


def get_opt_opt_state_params(
        conf: Conf,
        params: Union[PyTree, UDEProblem],
        schedule: optax.Schedule
):
    # Optimisers to choose from
    optimisers = {
        "adam": optax.adam(schedule),
        "adamw": optax.adamw(schedule),
        "adagrad": optax.adagrad(schedule),
        "sgd": optax.sgd(schedule),
    }

    if conf.optimiser not in optimisers.keys():
        raise ValueError(f"Unknown optimiser: {conf.optimiser}!")

    opt = optimisers[conf.optimiser]
    opt_state = opt.init(params)

    return opt, opt_state, params


def initialise_wandb(
        conf: Conf,
        model: Union[UDEProblem, eqx.nn.MLP],
):
    # Link to repo
    repo = git.Repo(search_parent_directories=True)
    # wandb.require("legacy-service")  # new wandb core
    # Initialise run
    wandb.init(
        project=f"TestUDE.snakemake.{conf.dynamical_model}.{conf.experiment_name}",
        group=f"{conf.training_setting}.{conf.shooting_mode}.{conf.curriculum_learning}",
        config=conf.__dict__,
        name='__'.join(field for field in conf.__str__().split('__')[1:]),  # dynamical_system info already in project
        settings=wandb.Settings(
            start_method="fork",
            git_commit=repo.head.object.hexsha,
            git_remote_url=repo.remotes.origin.url,
        ),
        mode="offline"
    )

    # Define shared metrics
    metrics = {
        "loss_train": "min",
        "loss_val": "min",
        "learning_rate": None,
    }

    # Add optional Hessian metrics -- both UDE and MLP
    if conf.compute_hessian_metrics:
        metrics = {**metrics, **hessian_metrics_wandb}
    # Add regularisation term if applied -- both UDE and MLP
    if conf.mlp_reg_strength > 0:
        metrics["regularisation_loss"] = "min"

    # Add UDE-only metrics
    if conf.training_setting == "UDE":
        metrics["max_grad_norm"] = "min"
        metrics["num_train_samples"] = None

        # neural network contribution/output regularisation
        if conf.output_reg > 0:
            metrics["output_regularisation_loss"] = "min"

        # End of training metrics (ADAM/BFGS)
        final_metrics = []
        for opt_num_epochs, opt_label in zip([conf.num_epochs_adam, conf.num_epochs_bfgs], ["ADAM", "BFGS"]):
            if opt_num_epochs > 0:
                final_metrics.extend(
                    [
                        f"final_loss_{dataset}_{opt_label}" for dataset in ["train", "val"]
                    ] +
                    [
                        f"final_{param_label}_{opt_label}"
                        for param_label in dynamic_param_labels[conf.dynamical_model]
                    ]
                )
        metrics = {
            **metrics,
            **{
                final_metric: None
                for final_metric in final_metrics
            }
        }


    # Define metrics in W&B
    for metric in metrics.keys():
        # if metric summary not specified
        if metrics[metric] is None:
            wandb.define_metric(metric)
        else:
            wandb.define_metric(metric, summary=metrics[metric])

    if isinstance(model, UDEProblem):
        mlp = model.func.mlp
    elif isinstance(model, eqx.nn.MLP):
        mlp = model
    else:
        raise ValueError(f"Unknown model type: {type(model)}!")
    # Define layer statistics (weight/bias value and grads)
    # from Fabian's NeuralCoarseGraining
    # https://github.com/frohlich-lab/NeuralCoarseGraining/blob/8c5cc42447b9b55cdf8b3bfc685388ba4bf6b751/ncg/training.py#L257
    for il, _ in enumerate(mlp.layers):
        for par_label, val_label in itt.product(('w', 'b'), ('vals', 'grads')):
            wandb.define_metric(
                f'layer{il}.{par_label}_{val_label}',
                summary='last',
            )


# Utilities to define epochs for logging and model saving
def enforce_minimum_spacing(arr: np.ndarray, min_dist: int) -> np.ndarray:
    """
    Filters an input array, arr, and keeps the original items that are at least min_dist apart.
    """
    prev = - np.inf  # ensure first element is always kept
    return np.array([prev := x for x in arr if x - prev >= min_dist])


def generate_log_epochs(n_epochs: int, num_samples: int, min_dist: int) -> np.ndarray:
    """
    Returns epochs regularly spaced in log10 space but no closer than min_dist
    to prevent running into filestream rate limit in W&B.
    """
    log_epochs = np.unique(
        np.logspace(0, np.log10(n_epochs), num=num_samples).astype(int)
    )
    return enforce_minimum_spacing(log_epochs, min_dist)


def get_save_epochs(num_epochs, num_models):
    if num_models <= 0 or num_epochs <= 0:
        return []
    if num_models == 1:
        return [num_epochs - 1]

    return [round(i * (num_epochs - 1) / (num_models - 1)) for i in range(num_models)]


# Utilities to make training step or compute gradient and hessian metrics
def make_grads_finite(grads):
    return jax.tree.map(
        lambda x: jnp.where(jnp.isfinite(x), x, jnp.zeros_like(x)),
        grads,
    )


def max_grad_norm(grads: PyTree) -> Array:
    # Flatten
    grads_flat, _ = jax.tree_util.tree_flatten(grads)
    # Compute L2 norm
    grad_norms = [jnp.linalg.norm(g) for g in grads_flat]
    # Return max norm
    return jnp.max(jnp.array(grad_norms))


def build_hessian_index_mask(model: UDEProblem, conf: Conf, filter_spec: PyTree):
    # Use precomputed ADAM filter_spec to initialise all-keeping mask_tree
    arrays_tree  = eqx.filter(model, filter_spec)
    mask_tree = jax.tree.map(lambda x: jnp.ones_like(x, dtype=bool), arrays_tree)

    state_dim = int(model.func.state_dimension)
    # Compute indices of parameters to exclude from Hessian computation (False)
    if conf.shooting_mode == "single":
        init_mask = jnp.concatenate(
            [
                jnp.zeros((state_dim,), dtype=bool),  # freeze physical state initials
                jnp.ones((conf.node_augmentation,),   dtype=bool),   # train augmented initials
                jnp.zeros((1,),  dtype=bool),   # freeze network contribution regularisation
            ],
            axis=0
        )
    elif conf.shooting_mode == "multiple":
        state_delim = int(model.func.state_delimiter)  # number of (state + augmented) variables
        n_windows = int(jnp.shape(model.initial_values)[0])

        # Freeze network contribution initial value for every window
        per_win_mask = jnp.concatenate(
            [jnp.ones((state_delim,), dtype=bool), jnp.array([False], dtype=bool)], axis=0
        )
        init_mask = jnp.tile(per_win_mask[None, :], (n_windows, 1))

        # Freeze the first window's state variables (not augmented variables)
        init_mask = init_mask.at[0, :state_dim].set(False)
    else:
        raise ValueError(f"Unknown shooting_mode: {conf.shooting_mode}")

    # Update mask_tree and ravel parameter & mask with same structure
    mask_tree = eqx.tree_at(lambda x: x.initial_values, mask_tree, init_mask)
    theta0, _   = ravel_pytree(arrays_tree)   # flattened params
    mask_vec, _ = ravel_pytree(mask_tree)     # flattened bools, aligned 1:1

    # Return indices to keep (active) for Hessian computation
    active_idx = jnp.nonzero(jnp.asarray(mask_vec, dtype=jnp.bool_))[0].astype(jnp.int32)
    return jnp.asarray(active_idx)


def get_hessian_metrics(
        hessian: Array,
        return_eigs_svs: bool = False,
        active_idx: Optional[Array] = None
) -> Union[tuple[dict, Array, Array], dict]:
    if active_idx is not None:
        unfrozen_hessian = hessian[jnp.ix_(active_idx, active_idx)]
    else:
        unfrozen_hessian = hessian
    # Compute metrics on the reduced Hessian
    hessian_trace = jnp.trace(unfrozen_hessian)
    hessian_diag = unfrozen_hessian * jnp.eye(unfrozen_hessian.shape[0])
    max_abs_diag = jnp.max(jnp.abs(hessian_diag))
    frob_diag = jnp.linalg.norm(hessian_diag)
    frob_off_diag = jnp.linalg.norm(unfrozen_hessian - hessian_diag)
    max_abs_off_diag = jnp.max(jnp.abs(unfrozen_hessian - hessian_diag))
    frob_ratio_off_to_diag = frob_off_diag / frob_diag

    # Compute singular values and eigenvalues
    singular_values = jnp.linalg.svd(
        unfrozen_hessian,
        hermitian=True,  # hermitian + real = symmetric
        compute_uv=False  # only return singular values
    )
    eigenvalues, _ = jnp.linalg.eigh(unfrozen_hessian)
    hessian_metrics = {
        "hessian_trace": hessian_trace,
        "max_abs_diagonal_element": max_abs_diag,
        "max_abs_off_diagonal_element": max_abs_off_diag,
        "frobenius_norm_diagonal": frob_diag,
        "frobenius_norm_off_diagonal": frob_off_diag,
        "ratio_frobenius_norm_offdiag_to_diag": frob_ratio_off_to_diag,
        "max_eigenvalue": jnp.max(eigenvalues),
        "min_eigenvalue": jnp.min(eigenvalues),
        "max_sv": jnp.max(singular_values),
        "min_sv": jnp.min(singular_values),
        "condition_number": jnp.max(singular_values)/jnp.min(singular_values),
    }
    return (hessian_metrics, eigenvalues, singular_values) if return_eigs_svs else hessian_metrics


# Metric handler to check whether to break training loop
# Tested (tests/test_training.py)
class MetricHandler:
    def __init__(self, patience=5):
        self.counter = 0
        self.patience = patience
        self.should_break = False

    def check_whether_should_break(self, metric):
        if not jnp.isfinite(metric):
            self.counter += 1
            if self.counter >= self.patience:  # fixed budget of patience
                print("Too many invalid values, breaking now.")
                self.should_break = True
        else:
            self.counter = 0
        return self.should_break


def freeze_ivs(ivs, state_dim):
    # Freeze initial values of state variables in first shooting window
    ivs = ivs.at[0, :state_dim].set(
        jax.lax.stop_gradient(ivs[0, :state_dim])
    )
    # Freeze network contribution regularisation initial value in all windows
    ivs = ivs.at[:, -1].set(
        jax.lax.stop_gradient(ivs[:, -1])
    )
    return ivs


# Utilities to compute solution and loss for different training schemes
def get_sol_and_loss_validation(
        prob: UDEProblem,
        u0: Array,
        t_save: SaveAt,
        t_span: List,
        adjoint: diffrax.AbstractAdjoint,
        loss_fn: Callable,
        data: Array,
        max_steps: int = 40000  # 2x as long for 2x timecourse
):
    sol = prob.solve_single_shooting_validation(
        diffrax.Tsit5(),
        u0,
        t_save,  # use provided t_save
        t_span,  # use provided t_span
        adjoint=adjoint,
        max_steps=max_steps
    )
    loss = loss_fn(
        solution=sol.ys,
        data=data,
    )
    return sol, loss


def get_sol_and_loss_single_shooting(
        prob: UDEProblem,
        adjoint: diffrax.AbstractAdjoint,
        loss_fn: Callable,
        data: Array,
        t_save_windows: Array,
        masks: Array,
        max_steps: int = 20000
):
    # Freeze state variables + network contribution regularisation and convert to 1D array
    ivs = freeze_ivs(prob.initial_values, prob.func.state_dimension)[0]

    sol = prob.solve_single_shooting(
        diffrax.Tsit5(),
        ivs,
        t_save_windows[0],
        adjoint,
        max_steps
    )
    operand = dict(solution=sol.ys, data=data, mask=masks.squeeze())
    return sol, jax.lax.cond(
        sol.result == diffrax.RESULTS.max_steps_reached,
        lambda _: jnp.nan,
        lambda op: loss_fn(**op),
        operand
    )


def loss_multiple_shooting(
        prob,
        loss_fn: Callable,
        sol,
        data,
        masks,
        last_valid_index,
):
    losses = jnp.array([
        loss_fn(
            solution=sol.ys[i, :, :prob.func.state_dimension],
            data=data,
            mask=masks[i]
        )
        for i in range(last_valid_index)
    ])
    return jnp.mean(losses)


def get_sol_and_loss_multiple_shooting(
        prob: UDEProblem,
        adjoint: diffrax.AbstractAdjoint,
        loss_fn: Callable,
        data: Array,
        t_save_windows: Array,
        masks: Array,
        last_valid_index: int,
        max_steps: int = 20000
):
    # Freeze state variables (first window) + network contribution regularisation (everywhere)
    ivs = freeze_ivs(prob.initial_values[:last_valid_index], prob.func.state_dimension)

    sol = prob.solve_multiple_shooting(
        diffrax.Tsit5(),
        ivs,
        t_save_windows[:last_valid_index],
        adjoint,
        max_steps
    )
    any_max_steps = jnp.any(sol.result == diffrax.RESULTS.max_steps_reached)
    loss_multiple_shooting_partial = partial(
        loss_multiple_shooting,
        prob=prob,
        loss_fn=loss_fn,
        last_valid_index=last_valid_index,
    )
    operand = dict(
        sol=sol,
        data=data,
        masks=masks,
    )
    return sol, jax.lax.cond(
        any_max_steps,
        lambda _: jnp.nan,
        lambda op: loss_multiple_shooting_partial(**op),
        operand
    )


def check_and_progress_curriculum(
        t_save_windows,
        masks,
        nn_epoch: int,
        cl_stage: int,
        model_save_steps_start: List,
        curriculum_params: dict
):
    advancing_cl_stage = False
    # If the epoch matches, update CL stage, inner timepoints and masks, and zero last set of initial values (window)
    if (nn_epoch != 0) and (nn_epoch in model_save_steps_start):
        print("----------------------------------")
        cl_stage = int(
            jnp.argmin(
                jnp.abs(jnp.array(model_save_steps_start) - nn_epoch)
            )
        ) + 1  # 1-counting
        t_save_windows = curriculum_params[cl_stage]["time"]
        masks = curriculum_params[cl_stage]["mask"]
        advancing_cl_stage = True
    return t_save_windows, masks, cl_stage, advancing_cl_stage


def prep_bfgs_training(model, zero_out_mlp, use_output_reg: bool):
    # Zero out MLP if needed (mechanistic fit only)
    if zero_out_mlp:
        mlp_replacement = set_weight_bias(model.func.mlp, 0.0, jax.random.PRNGKey(0))
    else:
        mlp_replacement = model.func.mlp

    # Set dropout behaviour to be deterministic and enable output/neural network contribution regularisation
    model_bfgs = model.replace_mlp(
        new_mlp=mlp_replacement,
        switch_to_inference=True,
        regularise_network=use_output_reg if (use_output_reg and not model.func.regularise_network) else None
    )
    return model_bfgs


def get_train_val_mse(
        model: UDEProblem,
        u0_train: jnp.ndarray,
        t_save_train: Union[SaveAt, Array],
        t_span_train: List,
        training_data: jnp.ndarray,
        u0_val: jnp.ndarray,
        t_save_val: Union[SaveAt, Array],
        t_span_val: List,
        validation_data: jnp.ndarray,
        loss_dict: Dict[str, Callable],
) -> Tuple[float, float]:
    """
    Computes both training and validation single-shooting MSE using the provided initial conditions (u0_train, u0_val).
    These losses do not include any regularisation terms.
    """
    params, static = eqx.partition(model, eqx.is_array)
    mse_train, _ = loss_dict["val_loss_fn"](
        prob_params=params,
        prob_static=static,
        u0=u0_train,
        t_save=t_save_train,
        t_span=t_span_train,
        data=training_data,
    )
    mse_val, _ = loss_dict["val_loss_fn"](
        prob_params=params,
        prob_static=static,
        u0=u0_val,
        t_save=t_save_val,
        t_span=t_span_val,
        data=validation_data
    )
    return mse_train, mse_val
