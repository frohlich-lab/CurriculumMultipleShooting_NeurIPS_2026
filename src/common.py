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
    mlp_layer_width: int = 5  # change back to 5
    mlp_depth: int = 1  # change back to 1
    init_scale: float = 0.1
    use_bias: bool = True
    use_final_bias: bool = True
    activation_fn_name: str = "relu"
    final_activation_fn_name: str = "identity"
    optimiser: str = "adam"
    # default max_lrate is 0.01 as in UDE paper
    max_lrate: float = 0.01  # absolute maximum learning rate (max in first schedule or in all schedules without decay)
    # default lrate_span is 1 = no range spanned
    lrate_span: float = 1e0  # ratio between max and min learning rates in a given schedule
    # default lrate_decay is 1 - no decay
    lrate_decay: float = 0.99  # if < 1, the learning rate decays between schedules
    # default warmup_fct = 0 as in UDE paper (no warmup)
    warmup_fct: float = 0.1  # fraction of schedule epochs to be used for warmup
    # default opt_steps creates a single schedule if N_EPOCHS = 2000
    opt_steps: int = 10  # Number of steps in the first schedule (they multiply each time in length by opt_mult)
    opt_mult: int = 2  # Multiplier for the number of steps in each schedule
    # they did not use early_stop in original UDE paper
    use_early_stop: bool = False
    reg_exp: float = 1.0
    reg_strength: float = 0.0  # no regularisation by default
    use_dropout: bool = False
    input_dropout_rate: float = 0.8
    hidden_dropout_rate: float = 0.5
    use_lookahead: bool = False
    lookahead_sync: int = 10
    lookahead_step_size: float = 0.5
    lookahead_reset_state: bool = False
    freeze_mechanistic_params_optx: bool = False
    # Extra options for UDEFunction
    node_augmentation: int = 0
    # add_interaction: bool = False
    dynamic_params_scale: str = "log10"
    save_plots: bool = False
    hessian_precondition: bool = False
    curriculum_learning: bool = True
    num_curriculum_steps: int = 5
    max_curriculum_epochs: Optional[int] = 2000
    multistart_deviation: float = 0.0  # default: ground truth parameters
    multistart_job: int = 0
    # mechanistic_fit_epochs: int = 0

    def __str__(self):
        """
        Return string representation of the configuration.

        Note: enables printing of Conf objects in fire
        """
        unwanted_fields = []
        if not self.use_dropout:  # if not using dropout, do not include dropout rates in run name
            unwanted_fields.extend(["input_dropout_rate", "hidden_dropout_rate"])
        values = (
            getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field not in unwanted_fields
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
