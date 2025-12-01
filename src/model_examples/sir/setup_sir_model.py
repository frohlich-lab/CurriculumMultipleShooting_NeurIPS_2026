import equinox as eqx
import jax
import jax.numpy as jnp

from src.model_training.base_losses import get_mse, get_masked_mse
from src.model_training.model_util import process_mechanistic_params
from src.model_examples.base_ude_function import BaseUDEFunction
from src.model_examples.data_loading import load_training_validation_csv
from jaxtyping import Array
from typing import Union


class SIRUDEFunction(BaseUDEFunction):
    """
    Class to evaluate right hand side (vector field) of an UDEModel

    Methods
    -------
    __call__(t, u, args)
        Evaluates the right hand side for input vector u, at time t
    """

    dynamic_parameters: Array
    mlp: Union[eqx.nn.MLP, eqx.Module]
    options: dict[str, Union[str, dict]]
    state_dimension: int
    state_delimiter: int
    regularise_network: bool

    def __call__(self, t, u, args):
        # Extract constant population size
        population_size = jax.lax.stop_gradient(self.options["extra_args"]["population_size"])

        # Get dynamic parameters on correct scale for ODE
        beta, gamma, delta = process_mechanistic_params(self.dynamic_parameters, self.options)
        s, i, r, t_ = u[:self.state_dimension]

        if self.state_delimiter > self.state_dimension:
            augmented_vars = u[self.state_dimension : self.mlp.layers[0].in_features]
            nn_q = self.mlp(jnp.array([s, i, r, *augmented_vars]))
        else:
            nn_q = self.mlp(jnp.array([s, i, r]))

        # QSIR model
        # TODO @Giacomo: discuss with Sebastian
        ds = - beta * s * i / population_size
        di = beta * s * i / population_size - gamma * i  - nn_q[0] * i / population_size
        dr = gamma * i + delta * t_
        dt_ = nn_q[0] * i / population_size  - delta * t_
        derivatives = jnp.array([ds, di, dr, dt_])

        if self.state_delimiter > self.state_dimension:
            # Add augmented variables
            derivatives = jnp.concatenate([derivatives, nn_q[1:]])  # default single output - Q: quarantine strength
        if self.regularise_network:
            # Add additional equation for regularisation term on network contribution
            d_nn_reg = jnp.array([
                jnp.linalg.norm(nn_q[0] * i / population_size)
            ]) # contribution is always the same in both di and dt
            derivatives = jnp.concatenate([derivatives, d_nn_reg])
        return derivatives


# TODO @GiacomoFabrini no magic numbers - add conf.node_augmentation and derive number of components for loss.
# LOSS AND GRAD FUNCTIONS
def sir_base_loss(solution: jnp.ndarray, data):
    infected, dead_recovered = data[:, 0], data[:, 1]
    # loss is sum of MSE on Infected and Recovered+Dead
    loss1 = get_mse(
        data=jnp.log(infected),
        pred=jnp.log(jnp.absolute(solution[:, 1] + solution[:, 3])),
    )
    loss2 = get_mse(
        data=jnp.log(dead_recovered),
        pred=jnp.log(jnp.absolute(solution[:, 2])),
    )
    # loss1 = jnp.mean((jnp.log(infected) - jnp.log(jnp.absolute(sol.ys[:, 1] + sol.ys[:, 3]))) ** 2)
    # loss2 = jnp.mean((jnp.log(dead_recovered) - jnp.log(jnp.absolute(sol.ys[:, 2]))) ** 2)
    return loss1 + loss2


def sir_base_loss_masked(solution: jnp.ndarray, data, mask: jnp.ndarray):
    infected, dead_recovered = data[:, 0], data[:, 1]
    # loss is sum of MSE on Infected and Recovered+Dead
    loss1 = get_masked_mse(
        data=jnp.log(infected),
        pred=jnp.log(jnp.absolute(solution[:, 1] + solution[:, 3])),
        mask=mask,
    )
    loss2 = get_masked_mse(
        data=jnp.log(dead_recovered),
        pred=jnp.log(jnp.absolute(solution[:, 2])),
        mask=mask,
    )
    # loss1 = jnp.mean((jnp.log(infected) - jnp.log(jnp.absolute(sol.ys[:, 1] + sol.ys[:, 3]))) ** 2)
    # loss2 = jnp.mean((jnp.log(dead_recovered) - jnp.log(jnp.absolute(sol.ys[:, 2]))) ** 2)
    return loss1 + loss2


# UTILS - DATA AND PLOTTING
def get_training_validation_data(country: str):
    """Load SIR training and validation data from CSV files."""
    if country == "Korea":
        return load_training_validation_csv(
            folder="sir",
            train_file="Korea_data_train.csv",
            validation_file="Korea_data_validation.csv",
            columns=["infected", "dead_recovered"],
            time_col="time",
        )
    elif country == "UK":
        return load_training_validation_csv(
            folder="sir",
            train_file="UK_data_train.csv",
            validation_file="UK_data_validation.csv",
            columns=["infected", "dead_recovered"],
            time_col="time",
        )
    else:
        raise ValueError(f"Unknown country: {country}")
