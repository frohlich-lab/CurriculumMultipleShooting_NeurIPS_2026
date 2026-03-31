from typing import Tuple, Union

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from src.model_examples.base_ude_function import BaseUDEFunction
from src.model_examples.data_loading import load_training_validation_csv
from src.model_training.base_losses import get_masked_mse, get_mse
from src.model_training.model_util import process_mechanistic_params


class LotkaVolterraUDEFunction(BaseUDEFunction):
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
    state_delimiter: int
    regularise_network: bool

    def __call__(self, t, u, args):
        # Get dynamic parameters on correct scale for ODE
        alpha, delta = process_mechanistic_params(self.dynamic_parameters, self.options)

        # u contains: state variables, node augmentation variables and regularisation term - latter is not NN input
        prey, predator = u[:self.state_dimension]
        nn_input = u[:self.state_delimiter]
        nn_out = self.mlp(nn_input)

        # Lotka-Volterra model
        d_prey = alpha * prey + nn_out[0]
        d_predator = nn_out[1] - delta * predator
        derivatives = jnp.array([d_prey, d_predator])

        if self.state_delimiter > self.state_dimension:
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
def get_training_validation_data(noise_level: int = 5, time_span: int = None):
    """Load Lotka-Volterra training and validation data from CSV files."""
    if time_span is not None:
        return load_training_validation_csv(
            folder="lotka_volterra",
            train_file=f"Train_data_time{time_span}.csv",
            validation_file=f"Validation_data_time{time_span}.csv",
            columns=["prey", "predator"],
            time_col="time",
        )

    return load_training_validation_csv(
        folder="lotka_volterra",
        train_file=f"Train_data_noise{noise_level}.csv",
        validation_file=f"Validation_data_noise{noise_level}.csv",
        columns=["prey", "predator"],
        time_col="time",
    )
