import dataclasses
import jax.numpy as jnp
import jax.nn as jnn
import jax
import numpy as np

from pathlib import Path
from jaxtyping import Array, PyTree
from typing import List, Optional, Union


# TODO @GiacomoFabrini - review these based on experiments!
unwanted_fields = [
    "dynamical_model",
    "final_activation_fn_name",
    "use_skip_connections",  # might need
    "use_dropout", "input_dropout_rate", "hidden_dropout_rate",  # might need
    "serialise_curriculum_models",
    "dynamic_params_lb", "dynamic_params_ub",
    "get_sindy_formula", "sindy_use_derivatives",
    "freeze_mechanistic_params_adam", "freeze_mechanistic_params_optx",  # might need
    "freeze_nn_adam", "freeze_nn_optx",  # might need
    "second_curriculum_learning",
    "save_plots",
]


@dataclasses.dataclass(frozen=True, repr=True, eq=True, order=True)
class Conf(dict):
    # Dynamical model to simulate
    dynamical_model: str = ''
    # Neural Network architecture, initialisation, settings and activations
    mlp_layer_width: int = 5  # change back to 5
    mlp_depth: int = 1  # change back to 1
    init_scale: float = 0.1
    use_bias: bool = True
    use_final_bias: bool = True
    activation_fn_name: str = "relu"
    final_activation_fn_name: str = "identity"
    use_skip_connections: bool = False
    # Dropout
    use_dropout: bool = False
    input_dropout_rate: float = 0.8
    hidden_dropout_rate: float = 0.5
    # Optimiser and learning rate schedule
    optimiser: str = "adam"
    max_lrate: float = 0.01  # absolute maximum learning rate (default max in UDE paper)
    lrate_span: float = 1e0  # ratio between max and min learning rates in a given schedule (default=1, min=max)
    lrate_decay: float = 0.99  # if < 1, the learning rate decays between schedules (default: no decay)
    warmup_fct: float = 0.1  # fraction of schedule epochs to be used for warmup (default: 10% warmup)
    opt_steps: int = 10  # Number of steps in the first schedule (they multiply each time in length by opt_mult)
    opt_mult: int = 2  # Multiplier for the number of steps in each schedule
    opt_sched_type: str = "cos"  # options: "cos", "lin", "exp"
    # Early-Stopping
    use_early_stop: bool = False  # default: False, not in use
    # Regularisation
    # MLP regularisation (L1/L2/network contribution regularisation)
    mlp_reg_exp: float = 1.0  # default: L1
    mlp_reg_strength: float = 0.0  # no regularisation by default
    # Output regularisation
    output_reg: float = 0.0  # no network contribution regularisation by default (UDE paper)
    # Simulation strategies: single/multiple-shooting, window-batching, curriculum learning
    shooting_mode: str = "single"
    shooting_window_length: int = 10
    shooting_iv_init: Optional[str] = "z"  # z: zeros, g: guess, ru: random uniform
    multiple_shooting_penalty: float = 0.0
    # batch_size: int = 64  # batch size for window-batching -- DISABLED
    # Curriculum Learning (growing fits)
    curriculum_learning: bool = False
    num_curriculum_steps: int = 5
    max_curriculum_epochs: Optional[int] = 2000
    serialise_curriculum_models: bool = False
    # Dynamic parameters: estimation scale, sampling, bounds
    dynamic_params_scale: str = "log10"  # can be linear, log10 or bounded tanh
    multistart_deviation: float = 0.0  # default: ground truth parameters
    multistart_job: int = 0
    dynamic_params_perturb_method: str = "su" # su=sample_uniform, agn=add_gaussian_noise
    dynamic_params_lb: float = 1e-3  # default lower bound
    dynamic_params_ub: float = 1e3  # default upper bound
    # UDE Augmentation
    node_augmentation: int = 0
    # Additional computations: Hessian, SINDy
    hessian_precondition: bool = False
    compute_hessian_metrics: bool = False
    get_sindy_formula: bool = False
    sindy_use_derivatives: bool = False
    # Mechanistic fit optimisation
    second_opt: str = "optx"  # alternative: "fides"
    # Freeze parts of the UDE model - TODO @GiacomoFabrini -- consider reducing to two params
    freeze_mechanistic_params_adam: bool = False
    freeze_mechanistic_params_optx: bool = False
    freeze_nn_adam: bool = False
    freeze_nn_optx: bool = False
    # Niche scenario settings: 2 curriculum learning stages (fit MLP, fit dynamic params)
    second_curriculum_learning: bool = False
    # Data pre-processing
    normalise_data: bool = False
    normalise_timepoints: bool = False
    # Plotting behaviour
    save_plots: bool = False
    # Training settings
    training_setting: str = None  # options: "UDE", "MLP"
    mlpexp_target: str = "RHS"  # options: "RHS", "interaction"
    experiment_name: str = None

    def __str__(self, serialise_model: bool = False) -> str:
        """
        Return string representation of the configuration. Removes unwanted fields.

        Note: enables printing of Conf objects in fire
        """

        # TODO @GiacomoFabrini - need to handle niche settings (loss landscape BFGS, second curriculum)

        values = (
            getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name not in unwanted_fields
        )
        return '__'.join(map(str, values))


@dataclasses.dataclass
class EarlyStoppingParams(dict):
    use_early_stopping: bool = True
    patience: int = 9
    min_improvement: float = 0


@dataclasses.dataclass
class TrainValArgs(dict):
    data: Array
    u0: Array
    t_save: Array
    t_span: List


# Flatten a tree to an array to allow gradient and Hessian functions to
# produce Array output
def flatten(v: PyTree):
    """
    Flatten a pytree to an array. This allows for vector gradient
    and Hessian functions to produce array output
    """
    def f(pt: PyTree) -> jnp.ndarray:
        leaves, _ = jax.tree_util.tree_flatten(pt)
        return jnp.concatenate([x.ravel() for x in leaves])

    out, pullback = jax.vjp(f, v)
    return out, lambda x: pullback(x)[0]


basedir: Path = Path(__file__).resolve().parent.parent  # Test_UDE, outside src
plots_dir = basedir / "plots"
results_dir = basedir / "results"
model_save_dir = basedir / "models"

default_attributes = [
    k
    for k, v in vars(Conf).items()
    if not k.startswith('__') and k not in unwanted_fields
]

# default_attributes = [
#     "dynamical_model",
#     "mlp_layer_width",
#     "mlp_depth",
#     "init_scale",
#     "use_bias",
#     "use_final_bias",
#     "activation_fn_name",
#     "optimiser",
#     "max_lrate",
#     "lrate_span",
#     "lrate_decay",
#     "warmup_fct",
#     "opt_steps",
#     "opt_mult",
#     "use_early_stop",
#     "mlp_reg_exp",
#     "mlp_reg_strength",
#     "output_reg",
#     "shooting_mode",
#     "shooting_window_length",
#     "shooting_iv_init",
#     "multiple_shooting_penalty",
#     "curriculum_learning",
#     "num_curriculum_steps",
#     "max_curriculum_epochs",
#     "dynamic_params_scale",
#     "multistart_deviation",
#     "multistart_job",
#     "dynamic_params_perturb_method",
#     "node_augmentation",
#     "hessian_precondition",
#     "compute_hessian_metrics",
#     "second_opt",
#     "normalise_data",
#     "normalise_timepoints",
#     "training_setting",
#     "mlpexp_target",
#     "experiment_name"
# ]

defaults = {
    x: f"{{{x}}}"
    for x in default_attributes
}

basic_template = "__".join(defaults.values())
TRAINING_RESULTS = str(results_dir / "{experiment_name}" / "{dynamical_model}" / "train" / (basic_template + ".csv"))
MERGED_TRAINING_RESULTS = str(results_dir / "{dynamical_model}" / "merged_results__{experiment_name}.csv")
AGGREGATION_PLOTS = str(plots_dir / "{experiment_name}" / "{dynamical_model}" / "{filename}.png")
PLOT_TYPES = [
    "final_loss_train_boxplot",
    "final_loss_val_boxplot",
    "corr_start_vs_final",
    "corr_train_vs_val"
]

def rbf(x):
    """
    Radial basis function activation function matching the one used in the original UDE paper.
    """
    return jnp.exp(-x**2)


# Dictionary of activation functions to choose from
activation_functions = {
    "identity": lambda x: x,  # identity = no activation function, default for final_activation
    "tanh": jnn.tanh,
    "relu": jnn.relu,
    "swish": jnn.swish,
    "elu": jnn.elu,
    "gelu": jnn.gelu,
    "rbf": rbf,
    "softplus": jnn.softplus,
    "squareplus": jnn.squareplus,
}

hessian_metrics_wandb = {
    "max_abs_diagonal_element": None,
    "max_abs_off_diagonal_element": None,
    "frobenius_norm_diagonal": None,
    "frobenius_norm_off_diagonal": None,
    "ratio_frobenius_norm_offdiag_to_diag": None,
    "max_eigenvalue": None,
    "min_eigenvalue": None,
    "max_sv": None,
    "min_sv": None,
    "condition_number": "min",
}

dynamic_param_labels = {
    "lotka_volterra": ["alpha", "delta"],
    "sir": ["beta", "gamma", "delta"],
}

# Ground truth dynamic parameters for models in use (LV, guesses only for SIR)
true_dynamic_parameters = {
    'lotka_volterra': jnp.array([1.3, 1.8]),
    'sir': jnp.array([0.15, 0.013, 0.01]),
}

# Starting conditions for models in use
u0s = {
    'lotka_volterra': jnp.array([0.44249296, 4.6280594]),
    'sir': jnp.array([51000000.0, 578, 24, 10]),
}

# State dimension (num state variables) for models in use
state_dims = {
    "lotka_volterra": 2,
    "sir": 4,
}

default_out_sizes = {
    "lotka_volterra": 2,
    "sir": 1,
}

def normalise(
        train_array: Union[Array, jnp.ndarray, np.array],
        val_array: Union[Array, jnp.ndarray, np.array],
        provide_u0: bool = False,
        u0: Union[Array, jnp.ndarray, np.array, None] = None
) -> tuple:
    """
    Normalise data or timepoint arrays by simply dividing by the max value in training set (in case of data, across all state
    variables and timepoints). Also return the normalisation scale.
    """
    # Cast to jnp.array
    train_array, val_array = jnp.array(train_array), jnp.array(val_array)
    if not provide_u0:
        return (train_array / jnp.max(train_array),
                val_array / jnp.max(train_array),
                jnp.max(train_array))
    else:
        return (train_array / jnp.max(train_array),
                val_array / jnp.max(train_array),
                jnp.array(u0) / jnp.max(train_array),
                jnp.max(train_array))
