import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import os
import pandas as pd

from src.model_training.model_util import regularise_mlp
from typing import Union


class UDEFunction(eqx.Module):
    """
    Class to evaluate right hand side (vector field) of an UDEModel

    Methods
    -------
    __call__(t, u, args)
        Evaluates the right hand side for input vector u, at time t
    """

    dynamic_parameters: jax.Array
    mlp: Union[eqx.nn.MLP, eqx.Module]
    options: dict[str, bool]

    def __init__(self, dynamic_parameters, mlp, options, **kwargs):
        super().__init__(**kwargs)
        self.dynamic_parameters = dynamic_parameters
        self.mlp = mlp
        self.options = options

    def __call__(self, t, u, args):
        u0 = args
        if self.options["dynamic_params_scale"] == "log10":
            beta, gamma, delta = jnp.power(10, self.dynamic_parameters)
        elif self.options["dynamic_params_scale"] == "linear":
            beta, gamma, delta = self.dynamic_parameters
        else:
            raise ValueError(f"Unknown dynamic parameter scale: {self.options['dynamic_params_scale']}!")
        s, i, r, t = u
        nn_t = self.mlp(jnp.array([s, i, r]))

        ds = - beta * s * i / u0[0]
        di = beta * s * i / u0[0] - gamma * i - nn_t[0] * i / u0[0]
        dr = gamma * i + delta * t
        dt = nn_t[0] * i / u0[0] - delta * t

        return jnp.array([ds, di, dr, dt])


# LOSS AND GRAD FUNCTIONS
def _compute_loss(sol, data):
    infected = data[:, 0]
    dead = data[:, 1]
    recovered = data[:, 2]
    loss1 = jnp.mean((jnp.log(infected) - jnp.log(jnp.absolute(sol.ys[:, 1] + sol.ys[:, 3]))) ** 2)
    loss2 = jnp.mean((jnp.log(jnp.absolute(recovered + dead)) - jnp.log(jnp.absolute(sol.ys[:, 2]))) ** 2)
    # loss is sum of MSE on Infected and Recovered+Dead
    return loss1 + loss2


@eqx.filter_value_and_grad
def compute_loss_grad(prob_params, prob_static, u0, t_save, t_span, data, reg_exp, reg_strength):
    prob = eqx.combine(prob_params, prob_static)
    sol = prob.solve(diffrax.Tsit5(), u0, t_save, t_span)
    return _compute_loss(sol, data) + regularise_mlp(prob.func.mlp, reg_exp, reg_strength)


@eqx.filter_jit
def compute_loss(prob, u0, t_save, t_span, data, reg_exp, reg_strength):
    sol = prob.solve(diffrax.Tsit5(), u0, t_save, t_span)
    return _compute_loss(sol, data) + regularise_mlp(prob.func.mlp, reg_exp, reg_strength)


@eqx.filter_jit
def compute_loss_optx(prob_params, args):
    prob_static, u0, t_save, t_span, data, reg_exp, reg_strength = args
    prob = eqx.combine(prob_params, prob_static)
    sol = prob.solve(diffrax.Tsit5(), u0, t_save, t_span)
    return _compute_loss(sol, data) + regularise_mlp(prob.func.mlp, reg_exp, reg_strength), None


# HESSIAN
@eqx.filter_jit
@jax.hessian
def compute_hess(
        prob_params, prob_static,
        u0, t_save_train, t_span_train, data, adjoint,
        reg_exp, reg_strength
):
    prob = eqx.combine(prob_params, prob_static)
    sol = prob.solve(diffrax.Tsit5(), u0, t_save_train, t_span_train, adjoint=adjoint)
    return _compute_loss(sol, data) + regularise_mlp(prob.func.mlp, reg_exp, reg_strength)


@eqx.filter_jit
@jax.hessian
def compute_hess_array(
        flat_p, unflatten_p, prob_static,
        u0, t_save_train, t_span_train, data, adjoint,
        reg_exp, reg_strength
):
    prob_params = unflatten_p(flat_p)
    prob = eqx.combine(prob_params, prob_static)
    sol = prob.solve(diffrax.Tsit5(), u0, t_save_train, t_span_train, adjoint=adjoint)
    return _compute_loss(sol, data) + regularise_mlp(prob.func.mlp, reg_exp, reg_strength)


# UTILS - DATA AND PLOTTING
def get_result_plot(prob, u0, t_save, t_span, data):
    """
    Transform observed and model prediction data into pd.DataFrames that 
    easily can be plotted with plotnine
    """
    # Model prediction
    sol = prob.solve(diffrax.Tsit5(), u0, t_save, t_span)
    df_pred1 = pd.DataFrame({"time": sol.ts,
                             "value": jnp.log(jnp.absolute(sol.ys[:, 1] + sol.ys[:, 3])),
                             "id": "Infected",
                             "datasource": "model"})
    df_pred2 = pd.DataFrame({"time": sol.ts,
                             "value": jnp.log(jnp.absolute(sol.ys[:, 2])),
                             "id": "Recovered + dead",
                             "datasource": "model"})
    df_predict = pd.concat([df_pred1, df_pred2], names=False)

    # Observed data
    infected, dead, recovered = data[:, 0], data[:, 1], data[:, 2]
    df_obs1 = pd.DataFrame({"time": sol.ts,
                            "value": jnp.log(infected),
                            "id": "Infected",
                            "datasource": "observed"})
    df_obs2 = pd.DataFrame({"time": sol.ts,
                            "value": jnp.log(jnp.absolute(recovered + dead)),
                            "id": "Recovered + dead",
                            "datasource": "observed"})
    df_obs = pd.concat([df_obs1, df_obs2], names=False)

    return df_predict, df_obs


def get_fit_dfs(prob, u0, t_save, t_span, data):
    # Model prediction
    sol = prob.solve(diffrax.Tsit5(), u0, t_save, t_span)
    df_pred = pd.DataFrame({"time": sol.ts,
                            "Infected": jnp.log(jnp.absolute(sol.ys[:, 1] + sol.ys[:, 3])),
                            "Recovered + dead": jnp.log(jnp.absolute(sol.ys[:, 2])),
                            })
    # Observed data
    infected, dead, recovered = data[:, 0], data[:, 1], data[:, 2]
    df_obs = pd.DataFrame({"time": sol.ts,
                           "Infected": jnp.log(infected),
                           "Recovered + dead": jnp.log(jnp.absolute(recovered + dead)),
                           })
    return df_pred, df_obs


def get_training_validation_data():
    # Read training and validation data
    # Training
    train_data_df = pd.read_csv(os.path.join("src", "data_dir", "sir", "Korea_data_train.csv"))
    train_data = jnp.array(train_data_df[["infected", "dead", "recovered"]].values)
    t_save_train = diffrax.SaveAt(ts=train_data_df["time"].values)
    t_span_train = [0.0, max(train_data_df["time"])]
    # Validation
    validation_data_df = pd.read_csv(os.path.join("src", "data_dir", "sir", "Korea_data_validation.csv"))
    validation_data = jnp.array(validation_data_df[["infected", "dead", "recovered"]].values)
    t_save_validation = diffrax.SaveAt(ts=validation_data_df["time"].values)
    t_span_validation = [0.0, max(validation_data_df["time"])]
    return train_data, t_save_train, t_span_train, validation_data, t_save_validation, t_span_validation
