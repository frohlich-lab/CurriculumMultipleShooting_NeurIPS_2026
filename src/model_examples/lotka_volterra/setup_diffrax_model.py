import diffrax
import equinox as eqx
import jax.numpy as jnp
import os
import pandas as pd

from src.model_training.model_util import get_mse, process_mechanistic_params
from jaxtyping import Array
from typing import Any, Union



class UDEFunction(eqx.Module):
    """
    Class to evaluate right hand side (vector field) of an UDEModel

    Methods
    -------
    __call__(t, u, args)
        Evaluates the right hand side for input vector u, at time t
    """

    dynamic_parameters: Array
    mlp: Union[eqx.nn.MLP, eqx.Module]
    options: dict[str, Union[bool, dict[str, Any]]]
    state_dimension: int
    regularise_network: bool

    def __init__(self, dynamic_parameters, mlp, options, state_dimension, regularise_network, **kwargs):
        super().__init__(**kwargs)
        self.dynamic_parameters = dynamic_parameters
        self.mlp = mlp
        self.options = options
        self.state_dimension = state_dimension
        self.regularise_network = regularise_network if regularise_network is not None else False

    def __call__(self, t, u, args):
        # Get dynamic parameters on correct scale for ODE
        alpha, delta = process_mechanistic_params(self.dynamic_parameters, self.options)
        # Unpack state variables
        prey, predator = u[:self.state_dimension]
        if len(u) > self.state_dimension + 1:
            nn_input = jnp.concatenate([u[:self.state_dimension], u[self.state_dimension+1:]])
        else:
            nn_input = u[:self.state_dimension]
        nn_out = self.mlp(nn_input)
        d_prey = alpha * prey + nn_out[0]
        d_predator = nn_out[1] - delta * predator
        if self.regularise_network:
            # Add additional equation for regularisation term on network contribution
            d_nn_reg = jnp.linalg.norm(nn_out[:self.state_dimension])
            derivatives = jnp.array([d_prey, d_predator, d_nn_reg])
        else:
            derivatives = jnp.array([d_prey, d_predator])
        if len(u) > self.state_dimension + 1:
            # Further add augmented variables
            derivatives = jnp.concatenate([derivatives, nn_out[self.state_dimension:]])
        return derivatives


def lv_base_loss(sol, data, state_dimension: int):
    return get_mse(pred=sol.ys[:, :state_dimension], data=data)


# UTILS - DATA AND PLOTTING
def get_training_validation_data():
    # Read training and validation data
    train_data_df = pd.read_csv(os.path.join("src", "data_dir", "lotka_volterra", "Train_data.csv"))
    train_data = jnp.array(train_data_df[["u1", "u2"]].values)
    t_save_train = diffrax.SaveAt(ts=train_data_df["t"].values)
    t_span_train = [0.0, max(train_data_df["t"])]
    validation_data_df = pd.read_csv(os.path.join("src", "data_dir", "lotka_volterra", "Validation_data.csv"))
    validation_data = jnp.array(validation_data_df[["u1", "u2"]].values)
    t_save_validation = diffrax.SaveAt(ts=validation_data_df["t"].values)
    t_span_validation = [0.0, max(validation_data_df["t"])]
    return train_data, t_save_train, t_span_train, validation_data, t_save_validation, t_span_validation


def get_result_plot(prob, u0, t_save, t_span, data):
    """
    Transform observed and model prediction data into pd.DataFrames that
    easily can be plotted with plotnine
    """
    # Model prediction
    u_pred = prob.solve_single_shooting(diffrax.Tsit5(), u0, t_save, t_span)
    df_pred1 = pd.DataFrame({"time": u_pred.ts,
                             "value": u_pred.ys[:, 0],
                             "id": "prey",
                             "datasource": "model"})
    df_pred2 = pd.DataFrame({"time": u_pred.ts,
                             "value": u_pred.ys[:, 1],
                             "id": "predator",
                             "datasource": "model"})
    df_predict = pd.concat([df_pred1, df_pred2], names=False)

    # Observed data
    df_obs1 = pd.DataFrame({"time": u_pred.ts,
                            "value": data[:, 0],
                            "id": "prey",
                            "datasource": "observed"})
    df_obs2 = pd.DataFrame({"time": u_pred.ts,
                            "value": data[:, 1],
                            "id": "predator",
                            "datasource": "observed"})
    df_obs = pd.concat([df_obs1, df_obs2], names=False)

    return df_predict, df_obs


def get_fit_dfs(prob, u0, t_save, t_span, data):
    # Model prediction
    u_pred = prob.solve_single_shooting(diffrax.Tsit5(), u0, t_save, t_span)
    df_pred = pd.DataFrame({"time": u_pred.ts,
                            "prey": u_pred.ys[:, 0],
                            "predator": u_pred.ys[:, 1],
                            })
    df_obs = pd.DataFrame({"time": u_pred.ts,
                           "prey": data[:, 0],
                           "predator": data[:, 1],
                           })
    return df_pred, df_obs
