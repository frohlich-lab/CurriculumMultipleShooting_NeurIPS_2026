import equinox as eqx
import jax.numpy as jnp

from src.model_training.model_util import (
    get_mse,
    get_masked_mse,
    process_mechanistic_params,
)
from src.model_examples.base import BaseUDEFunction
from src.model_examples.utils import load_training_validation_csv
from jaxtyping import Array
from typing import Optional, Union


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
    options: dict[str, bool]
    state_dimension: int
    state_delimiter: int
    regularise_network: bool

    def __call__(self, t, u, args):
        u0 = args
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
        ds = - beta * s * i / u0[0]
        di = beta * s * i / u0[0] - gamma * i  - nn_q[0] * i / u0[0]
        dr = gamma * i + delta * t_
        dt_ = nn_q[0] * i / u0[0]  - delta * t_
        derivatives = jnp.array([ds, di, dr, dt_])

        if self.state_delimiter > self.state_dimension:
            # Add augmented variables
            derivatives = jnp.concatenate([derivatives, nn_q[1:]])  # default single output - Q: quarantine strength
        if self.regularise_network:
            # Add additional equation for regularisation term on network contribution
            d_nn_reg = jnp.array([
                jnp.linalg.norm(nn_q[0] * i / u0[0])
            ]) # contribution is always the same in both di and dt
            derivatives = jnp.concatenate([derivatives, d_nn_reg])
        return derivatives


# TODO @GiacomoFabrini no magic numbers - add conf.node_augmentation and derive number of components for loss.
# LOSS AND GRAD FUNCTIONS
def sir_base_loss(solution: jnp.ndarray, data):
    infected, dead, recovered = data[:, 0], data[:, 1], data[:, 2]
    # loss is sum of MSE on Infected and Recovered+Dead
    loss1 = get_mse(
        data=jnp.log(infected),
        pred=jnp.log(jnp.absolute(solution[:, 1] + solution[:, 3])),
    )
    loss2 = get_mse(
        data=jnp.log(recovered + dead),
        pred=jnp.log(jnp.absolute(solution[:, 2])),
    )
    # loss1 = jnp.mean((jnp.log(infected) - jnp.log(jnp.absolute(sol.ys[:, 1] + sol.ys[:, 3]))) ** 2)
    # loss2 = jnp.mean((jnp.log(recovered + dead) - jnp.log(jnp.absolute(sol.ys[:, 2]))) ** 2)
    return loss1 + loss2


def sir_base_loss_masked(solution: jnp.ndarray, data, mask: jnp.ndarray):
    infected, dead, recovered = data[:, 0], data[:, 1], data[:, 2]
    # loss is sum of MSE on Infected and Recovered+Dead
    loss1 = get_masked_mse(
        data=jnp.log(infected),
        pred=jnp.log(jnp.absolute(solution[:, 1] + solution[:, 3])),
        mask=mask,
    )
    loss2 = get_masked_mse(
        data=jnp.log(recovered + dead),
        pred=jnp.log(jnp.absolute(solution[:, 2])),
        mask=mask,
    )
    # loss1 = jnp.mean((jnp.log(infected) - jnp.log(jnp.absolute(sol.ys[:, 1] + sol.ys[:, 3]))) ** 2)
    # loss2 = jnp.mean((jnp.log(recovered + dead) - jnp.log(jnp.absolute(sol.ys[:, 2]))) ** 2)
    return loss1 + loss2


# UTILS - DATA AND PLOTTING
def get_training_validation_data():
    """Load SIR training and validation data from CSV files."""
    return load_training_validation_csv(
        folder="sir",
        train_file="Korea_data_train.csv",
        validation_file="Korea_data_validation.csv",
        columns=["infected", "dead", "recovered"],
        time_col="time",
    )
