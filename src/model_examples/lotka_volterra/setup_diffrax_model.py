import equinox as eqx
import jax.numpy as jnp

from src.model_training.model_util import (
    get_mse,
    get_masked_mse,
    process_mechanistic_params,
)
from src.model_examples.base import BaseUDEFunction
from src.model_examples.utils import (
    load_training_validation_csv,
    get_result_plot_generic,
)
from jaxtyping import Array
from typing import Optional, Tuple, Union



class UDEFunction(BaseUDEFunction):
    """
    Class to evaluate right hand side (vector field) of an UDEModel

    Methods
    -------
    __call__(t, u, args)
        Evaluates the right hand side for input vector u, at time t
    """

    dynamic_parameters: Array
    mlp: Union[eqx.nn.MLP, eqx.Module]
    options: dict[str, Union[bool, dict[str, Tuple]]]
    state_dimension: int
    regularise_network: bool

    def __call__(self, t, u, args):
        # Get dynamic parameters on correct scale for ODE
        alpha, delta = process_mechanistic_params(self.dynamic_parameters, self.options)

        # u contains: state variables, node augmentation variables and regularisation term - latter is not NN input
        prey, predator = u[:self.state_dimension]
        nn_input = u[:self.mlp.layers[0].in_features]
        nn_out = self.mlp(nn_input)

        # Lotka-Volterra model
        d_prey = alpha * prey + nn_out[0]
        d_predator = nn_out[1] - delta * predator
        derivatives = jnp.array([d_prey, d_predator])

        if self.mlp.layers[0].in_features > self.state_dimension:
            # Add augmented variables
            derivatives = jnp.concatenate([derivatives, nn_out[self.state_dimension:]])
        if self.regularise_network:
            # Add additional equation for regularisation term on network contribution
            d_nn_reg = jnp.array([
                jnp.linalg.norm(nn_out[:self.state_dimension])
            ])
            derivatives = jnp.concatenate([derivatives, d_nn_reg])
        return derivatives


def lv_base_loss(solution: jnp.ndarray, data, state_dimension: int):
    return get_mse(pred=solution[:, :state_dimension], data=data)


def lv_base_loss_masked(solution: jnp.ndarray, data, state_dimension: int, mask: jnp.ndarray):
    return get_masked_mse(pred=solution[:, :state_dimension], data=data, mask=mask)


# UTILS - DATA AND PLOTTING
def get_training_validation_data():
    """Load Lotka-Volterra training and validation data from CSV files."""
    return load_training_validation_csv(
        folder="lotka_volterra",
        train_file="Train_data.csv",
        validation_file="Validation_data.csv",
        columns=["u1", "u2"],
        time_col="t",
    )


def get_result_plot(prob, u0, t_save, t_span, data):
    """Wrapper around :func:`get_result_plot_generic` for Lotka-Volterra."""

    def _pred_transform(ys):
        return ys[:, :2]

    def _data_transform(d):
        return d

    labels = ["prey", "predator"]

    return get_result_plot_generic(
        prob,
        u0,
        t_save,
        t_span,
        data,
        _pred_transform,
        _data_transform,
        labels,
    )
