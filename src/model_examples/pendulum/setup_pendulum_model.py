import equinox as eqx
import jax.numpy as jnp

from src.model_training.base_losses import get_mse, get_masked_mse
from src.model_examples.base_ude_function import BaseUDEFunction
from src.model_examples.data_loading import load_training_validation_csv
from jaxtyping import Array
from typing import Tuple, Union


class PendulumNODEFunction(BaseUDEFunction):
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
        # Unpack state variables
        theta1, theta2, omega1, omega2 = u[:self.state_dimension]

        # Feed state variables + NODE augmentation to neural network
        nn_input = u[:self.state_delimiter]

        # Get omega1, omega2 predictions
        nn_out = self.mlp(nn_input)

        # Double-pendulum 2nd order model
        d_theta1 = omega1
        d_theta2 = omega2
        d_omega1 = nn_out[0]
        d_omega2 = nn_out[1]

        derivatives = jnp.array([d_theta1, d_theta2, d_omega1, d_omega2])

        if self.state_delimiter > self.state_dimension:
            # Add augmented variables
            derivatives = jnp.concatenate([derivatives, nn_out[2:]])
        if self.regularise_network:
            # Add additional equation for regularisation term on network contribution
            d_nn_reg = jnp.array([
                jnp.linalg.norm(nn_out[:self.state_dimension])
            ])
            derivatives = jnp.concatenate([derivatives, d_nn_reg])
        return derivatives


def pendulum_base_loss(solution: jnp.ndarray, data, state_dimension: int):
    return get_mse(pred=solution[:, :state_dimension], data=data)


def pendulum_base_loss_masked(solution: jnp.ndarray, data, state_dimension: int, mask: jnp.ndarray):
    return get_masked_mse(pred=solution[:, :state_dimension], data=data, mask=mask)


# UTILS - DATA AND PLOTTING
def get_training_validation_data():
    """Load Lotka-Volterra training and validation data from CSV files."""
    return load_training_validation_csv(
        folder="pendulum",
        train_file="Train_data.csv",
        validation_file="Validation_data.csv",
        columns=["theta1", "theta2", "omega1", "omega2"],
        time_col="time",
    )
