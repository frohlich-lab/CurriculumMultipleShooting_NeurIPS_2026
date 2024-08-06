import jax.numpy as jnp
import diffrax
import equinox as eqx
import jax
import os
import pandas as pd

from src.model_training.model_util import regularise_mlp
from jaxtyping import Array
from typing import Union


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
    options: dict[str, bool]

    def __init__(self, dynamic_parameters, mlp, options, **kwargs):
        super().__init__(**kwargs)
        self.dynamic_parameters = dynamic_parameters
        self.mlp = mlp
        self.options = options

    def __call__(self, t, u, args):
        # Unpack dynamic parameters (learnable)
        if self.options["dynamic_params_scale"] == "log10":
            alpha, delta = jnp.power(10, self.dynamic_parameters)
        elif self.options["dynamic_params_scale"] == "linear":
            alpha, delta = self.dynamic_parameters
        else:
            raise ValueError(f"Unknown dynamic parameter scale: {self.options['dynamic_params_scale']}!")
        # Unpack state variables
        prey, predator = u[0], u[1]
        # if self.options["add_interaction"]:
        #     nn_input = jnp.concatenate([u, jnp.array([prey * predator])])
        # else:
        #     nn_input = u
        nn_input = u
        nn_out = self.mlp(nn_input)
        d_prey = alpha * prey + nn_out[0]
        d_predator = nn_out[1] - delta * predator
        derivatives = jnp.array([d_prey, d_predator])
        if len(u) > 2:
            derivatives = jnp.concatenate([derivatives, nn_out[2:]])
        return derivatives


# LOSS AND GRAD
@eqx.filter_value_and_grad
def compute_loss_grad(prob_params, prob_static, u0, t_save, t_span, data, reg_exp, reg_strength):
    prob = eqx.combine(prob_params, prob_static)
    u_pred = prob.solve(diffrax.Tsit5(), u0, t_save, t_span)
    return jnp.mean((u_pred.ys[:, :2] - data) ** 2) + regularise_mlp(prob.func.mlp, reg_exp, reg_strength)


# TODO @GiacomoFabrini no magic numbers - add conf.node_augmentation and derive number of components for loss.
#  Same for all functions on this page! Need to adapt SIR as well!
@eqx.filter_jit
def compute_loss(prob, u0, t_save, t_span, data, reg_exp, reg_strength):
    u_pred = prob.solve(diffrax.Tsit5(), u0, t_save, t_span)
    return jnp.mean((u_pred.ys[:, :2] - data) ** 2) + regularise_mlp(prob.func.mlp, reg_exp, reg_strength)


@eqx.filter_jit
def compute_loss_optx(prob_params, args):
    prob_static, u0, t_save, t_span, data, reg_exp, reg_strength = args
    prob = eqx.combine(prob_params, prob_static)
    u_pred = prob.solve(diffrax.Tsit5(), u0, t_save, t_span)
    return jnp.mean((u_pred.ys[:, :2] - data) ** 2) + regularise_mlp(prob.func.mlp, reg_exp, reg_strength), None


# HESSIAN
@eqx.filter_jit
@jax.hessian
def compute_hess(prob_params, prob_static, u0, t_save, t_span, data, adjoint):
    prob = eqx.combine(prob_params, prob_static)
    u_pred = prob.solve(diffrax.Tsit5(), u0, t_save, t_span,
                        adjoint=adjoint)
    return jnp.mean((u_pred.ys[:, :2] - data) ** 2)


@eqx.filter_jit
@jax.hessian
def compute_hess_array(flat_p, unflatten_p, prob_static, u0, t_save, t_span, data, adjoint, reg_exp, reg_strength):
    prob_params = unflatten_p(flat_p)
    prob = eqx.combine(prob_params, prob_static)
    u_pred = prob.solve(diffrax.Tsit5(), u0, t_save, t_span,
                        adjoint=adjoint)
    return jnp.mean((u_pred.ys[:, :2] - data) ** 2) + regularise_mlp(prob.func.mlp, reg_exp, reg_strength)


# UTILS - DATA AND PLOTTING
def get_training_validation_data():
    # Read training and validation data
    train_data_df = pd.read_csv(os.path.join("src", "data_dir", "lotka_volterra", "Train_data_GF_63_noise0.05.csv"))
    train_data = jnp.array(train_data_df[["u1", "u2"]].values)
    t_save_train = diffrax.SaveAt(ts=train_data_df["t"].values)
    t_span_train = [0.0, max(train_data_df["t"])]
    validation_data_df = pd.read_csv(os.path.join("src", "data_dir", "lotka_volterra", "Validation_data_GF_63_noise0.05.csv"))
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
    u_pred = prob.solve(diffrax.Tsit5(), u0, t_save, t_span)
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
    u_pred = prob.solve(diffrax.Tsit5(), u0, t_save, t_span)
    df_pred = pd.DataFrame({"time": u_pred.ts,
                            "prey": u_pred.ys[:, 0],
                            "predator": u_pred.ys[:, 1],
                            })
    df_obs = pd.DataFrame({"time": u_pred.ts,
                           "prey": data[:, 0],
                           "predator": data[:, 1],
                           })
    return df_pred, df_obs
