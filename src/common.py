import dataclasses
import jax.numpy as jnp
import jax.nn as jnn
import jax

from pathlib import Path
from jaxtyping import PyTree
from typing import Optional


@dataclasses.dataclass(frozen=True, repr=True, eq=True, order=True)
class Conf(dict):
    dynamical_model: str = ''
    # Neural Network architecture, initialisation, settings and activations
    mlp_layer_width: int = 5  # change back to 5
    mlp_depth: int = 1  # change back to 1
    init_scale: float = 0.1
    use_bias: bool = True
    use_final_bias: bool = True
    activation_fn_name: str = "relu"
    final_activation_fn_name: str = "identity"
    # Optimiser and learning rate schedule
    optimiser: str = "adam"
    # default max_lrate is 0.01 as in UDE paper
    max_lrate: float = 0.01  # absolute maximum learning rate
    lrate_span: float = 1e0  # ratio between max and min learning rates in a given schedule (default=1, min=max)
    lrate_decay: float = 0.99  # if < 1, the learning rate decays between schedules (default: no decay)
    warmup_fct: float = 0.1  # fraction of schedule epochs to be used for warmup (default: 10% warmup)
    opt_steps: int = 10  # Number of steps in the first schedule (they multiply each time in length by opt_mult)
    opt_mult: int = 2  # Multiplier for the number of steps in each schedule
    # Early Stopping
    use_early_stop: bool = False  # default: False, not in use
    # MLP regularisation (L1/L2/network contribution regularisation)
    mlp_reg_exp: float = 1.0  # default: L1
    mlp_reg_strength: float = 0.0  # no regularisation by default
    output_reg: float = 0.0  # no network contribution regularisation by default (UDE paper)
    use_dropout: bool = False
    input_dropout_rate: float = 0.8
    hidden_dropout_rate: float = 0.5
    use_lookahead: bool = False
    lookahead_sync: int = 10
    lookahead_step_size: float = 0.5
    lookahead_reset_state: bool = False
    # Freeze parts of the UDE model - TODO @GiacomoFabrini -- consider reducing to two params
    freeze_mechanistic_params_adam: bool = False
    freeze_mechanistic_params_optx: bool = False
    freeze_nn_adam: bool = False
    freeze_nn_optx: bool = False
    # Extra options for UDEFunction
    node_augmentation: int = 0
    dynamic_params_scale: str = "log10"
    save_plots: bool = False
    hessian_precondition: bool = False
    compute_hessian_metrics: bool = False
    # Curriculum Learning (growing fits)
    curriculum_learning: bool = True
    num_curriculum_steps: int = 5
    max_curriculum_epochs: Optional[int] = 2000
    serialise_curriculum_models: bool = False
    # Dynamic parameter sampling
    multistart_deviation: float = 0.0  # default: ground truth parameters
    multistart_job: int = 0
    dynamic_params_perturb_method: str = "sample_uniform"
    # mechanistic_fit_epochs: int = 0
    training_setting: str = None  # options: "UDE", "MLP"
    mlpexp_target: str = "RHS"  # options: "RHS", "interaction"
    curriculum_loss_landscape_BFGS: bool = False
    freeze_mechanistic_params_landscape_BFGS: bool = False
    use_fides: bool = False
    experiment_name: str = None
    get_sindy_formula: bool = False
    sindy_use_derivatives: bool = False

    def __str__(self):
        """
        Return string representation of the configuration.

        Note: enables printing of Conf objects in fire
        """
        unwanted_fields = [
            "experiment_name", "serialise_curriculum_models",
            "get_sindy_formula", "sindy_use_derivatives"
        ]
        if not self.use_dropout:  # if not using dropout, do not include dropout rates in run name
            unwanted_fields.extend(["input_dropout_rate", "hidden_dropout_rate"])
        if not self.use_lookahead:  # if not using LookAhead, do not include related params
            unwanted_fields.extend(["lookahead_sync", "lookahead_step_size", "lookahead_reset_state"])
        if not self.curriculum_learning:  # if not using curriculum learning, do not include number of steps/epochs
            unwanted_fields.extend(["num_curriculum_steps", "max_curriculum_epochs"])
        if self.mlp_reg_strength == 0:
            unwanted_fields.extend(["mlp_reg_exp", "mlp_reg_strength"])

        if self.training_setting != "MLP_experiment":  # only include the kind of target for MLP experiments
            unwanted_fields.extend(["mlpexp_target"])
        else:  # remove UDE-specific params in the case of MLP experiments
            unwanted_fields.extend(
                ["node_augmentation", "freeze_mechanistic_params_adam", "freeze_mechanistic_params_optx"]
            )
        # Handle BFGS loss landscape-specific parameters
        if not self.curriculum_loss_landscape_BFGS:
            unwanted_fields.extend(["freeze_mechanistic_params_landscape_BFGS"])
        else:  # for BFGS baseline, no need to report NN optimiser and related hyperparams, nor hessian
            unwanted_fields.extend(
                [
                    "optimiser", "max_lrate", "lrate_span", "lrate_decay", "warmup_fct", "opt_steps", "opt_mult",
                    "use_early_stop",
                    "freeze_mechanistic_params_adam", "freeze_mechanistic_params_optx",
                    "hessian_precondition", "compute_hessian_metrics",
                ]
            )

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


basedir: Path = Path(__file__).resolve().parent
plots_dir = basedir / "plots"
results_dir = basedir / "results"
model_save_dir = basedir / "models"


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
    "sir": ['beta', 'gamma', 'delta'],
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
