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
    "init_scale", "use_bias", "use_final_bias", "final_activation_fn_name",
    "use_dropout", "input_dropout_rate", "hidden_dropout_rate",  # might need
    "lrate_span", "lrate_decay", "warmup_fct", "opt_steps", "opt_mult", "opt_sched_type",
    "restart_opt_curriculum",
    "dynamic_params_lb", "dynamic_params_ub",
    "freeze_mechanistic_params_adam", "freeze_mechanistic_params_optx",  # might need
    "freeze_nn_adam", "freeze_nn_optx",  # might need
    "save_plots",
    "experiment_name",
    "num_starts",
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
    # Dropout
    use_dropout: bool = False
    input_dropout_rate: float = 0.8
    hidden_dropout_rate: float = 0.5
    # Optimiser and learning rate schedule
    training_scheme: str = "single"
    optimiser: str = "adam"
    num_epochs_adam: int = 4000  # number of epochs for Adam optimiser
    # default: constant learning rate (no schedule)
    max_lrate: float = 0.01  # absolute maximum learning rate (default max in UDE paper)
    lrate_span: float = 1  # ratio between max and min learning rates in a given schedule (default=1, min=max)
    lrate_decay: float = 1  # if < 1, the learning rate decays between schedules (default: no decay)
    warmup_fct: float = 0  # fraction of schedule epochs to be used for warmup (None for constant learning rate)
    opt_steps: int = 0  # Number of steps in the first schedule (they multiply each time in length by opt_mult)
    opt_mult: int = 0  # Multiplier for the number of steps in each schedule
    opt_sched_type: str = "lin"  # options: "cos", "lin", "exp"
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
    # Curriculum Learning (growing fits)
    curriculum_learning: bool = False
    num_curriculum_steps: int = 5
    max_curriculum_epochs: Optional[int] = 2000
    restart_opt_curriculum: bool = False  # restart optimiser at each curriculum step
    # Dynamic parameters: estimation scale, sampling, bounds
    dynamic_params_scale: str = "log10"  # can be linear, log10 or bounded tanh
    multistart_deviation: float = 0.0  # default: ground truth parameters
    multistart_job: int = 0
    dynamic_params_perturb_method: str = "lh" # lh=latin_hypercube, su=sample_uniform, agn=add_gaussian_noise
    dynamic_params_lb: float = 1e-3  # default lower bound
    dynamic_params_ub: float = 1e3  # default upper bound
    # UDE Augmentation
    node_augmentation: int = 0
    # Hessian metrics
    compute_hessian_metrics: bool = False
    # Mechanistic fit optimisation
    second_opt: str = "optx"  # alternative: "fides"
    num_epochs_bfgs : int = 0  # number of epochs for second optimiser (BFGS, fides/optimistix)
    # Freeze parts of the UDE model - TODO @GiacomoFabrini -- consider reducing to two params
    freeze_mechanistic_params_adam: bool = False
    freeze_mechanistic_params_optx: bool = False
    freeze_nn_adam: bool = False
    freeze_nn_optx: bool = False
    # Data pre-processing
    normalise_data: bool = False
    normalise_timepoints: bool = False
    # Plotting behaviour
    save_plots: bool = False
    # Training settings
    training_setting: str = None  # options: "UDE", "MLP"
    mlpexp_target: str = "None"  # options: "RHS", "interaction" (for old MLP experiments), "None" (for NODEs/UDEs)
    save_landscape_models: bool = False
    experiment_name: str = None
    noise_level: Optional[int] = None
    num_starts: int = 100

    def __str__(self) -> str:
        """
        Return string representation of the configuration. Removes unwanted fields.

        Note: enables printing of Conf objects in fire
        """
        values = (
            getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name not in unwanted_fields
        )
        return '__'.join(map(str, values))


# Define attributes for Linear Scans
scan_attributes = [
    "mlp_layer_width",
    "mlp_depth",
    "max_lrate",
    "mlp_reg_strength",
    "output_reg",
    "shooting_window_length",
    "multiple_shooting_penalty",
    "num_curriculum_steps",
    "max_curriculum_epochs",
    "node_augmentation",
]


@dataclasses.dataclass
class TrainValArgs(dict):
    data: Array
    u0: Array
    t_save: Array
    t_span: List


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
data_dir = basedir / "src" / "data_dir"

default_attributes = [
    k
    for k, v in vars(Conf).items()
    if not k.startswith('__') and k not in unwanted_fields
]

defaults = {
    x: f"{{{x:.6f}}}" if isinstance(x, float) else f"{{{x}}}"
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
OUTPUT_HDF5 = str(results_dir / "{experiment_name}" / "{dynamical_model}" / "{experiment_name}__results.hdf5")

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
    "sirKorea": ["beta", "gamma", "delta"],
    "sirUK": ["beta", "gamma", "delta"],
    "pendulum": [],  # pure NODE
}

# Ground truth dynamic parameters for models in use (LV, guesses only for SIR)
true_dynamic_parameters = {
    'lotka_volterra': jnp.array([1.3, 1.8]),
    'sirKorea': jnp.array([0.15, 0.013, 0.01]),
    'sirUK': jnp.array([0.15, 0.013, 0.01]),
    "pendulum": jnp.array([]),
}

# Starting conditions for models in use
u0s = {
    'lotka_volterra': jnp.array([0.44249296, 4.6280594]),
    'sirKorea': jnp.array([51000000.0, 578.0-10.0, 24.0, 10.0]),
    'sirUK': jnp.array([66000000.0, 772.0-10.0, 26.0, 10.0]),
    'pendulum': jnp.array([-1.1336150169372559, -1.3008047342300415, -0.5385388135910034, -6.1550583839416495]),  # 1st train datapoints
}

# State dimension (num state variables) for models in use
state_dims = {
    "lotka_volterra": 2,
    "sirKorea": 4,
    "sirUK": 4,
    "pendulum": 4,
}

default_out_sizes = {
    "lotka_volterra": 2,
    "sirKorea": 1,
    "sirUK": 1,
    "pendulum": 2,
}

max_iters_diffrax = {
    "lotka_volterra": 20000,
    "sirKorea": 100000,
    "sirUK": 100000,
    "pendulum": 20000,
}

def normalise(
        train_array: Union[Array, jnp.ndarray, np.array],
        val_array: Union[Array, jnp.ndarray, np.array],
        return_u0: bool = False,
        u0: Union[Array, jnp.ndarray, np.array, None] = None
) -> tuple:
    """
    Normalise data or timepoint arrays with respect to the max value in training set (in case of data, across all state
    variables and timepoints). If return_u0, also returns rescaled initial value. Also returns the normalisation scale.
    """
    # Cast to jnp.array
    train_array, val_array = jnp.array(train_array), jnp.array(val_array)
    if not return_u0:
        return (train_array / jnp.max(train_array),
                val_array / jnp.max(train_array),
                jnp.max(train_array))
    else:
        return (train_array / jnp.max(train_array),
                val_array / jnp.max(train_array),
                jnp.array(u0) / jnp.max(train_array),
                jnp.max(train_array))

def format_combos(v):
    if isinstance(v, float):
        return f"{{{v:.6f}}}"
    else:
        return v
    
def format_param_combos(param_combos):
    return [
        {k: format_combos(v) for k, v in cfg.items()}
        for cfg in param_combos
    ]