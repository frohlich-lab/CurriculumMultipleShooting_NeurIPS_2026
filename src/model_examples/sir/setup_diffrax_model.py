import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import os
import pandas as pd

from src.model_training.model_util import regularise_mlp, regularise_mlp_contribution, get_mse, process_mechanistic_params
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
        u0 = args
        # Get dynamic parameters on correct scale for ODE
        beta, gamma, delta = process_mechanistic_params(self.dynamic_parameters, self.options)
        s, i, r, t_ = u[:self.state_dimension]
        if len(u) > self.state_dimension + 1:
            augmented_vars = u[self.state_dimension+1:]  # exclude evolution of 2-norm of network contribution
            nn_q = self.mlp(jnp.array([s, i, r, *augmented_vars]))
        else:
            nn_q = self.mlp(jnp.array([s, i, r]))

        # TODO @Giacomo: discuss with Sebastian
        ds = - beta * s * i / u0[0]
        di = beta * s * i / u0[0] - gamma * i  - nn_q[0] * i / u0[0]
        dr = gamma * i + delta * t_
        dt_ = nn_q[0] * i / u0[0]  - delta * t_


        if self.regularise_network:
            # Add additional equation for regularisation term on network contribution
            d_nn_reg = jnp.linalg.norm(nn_q[0] * i / u0[0])  # contribution is always the same in both di and dt
            derivatives = jnp.array([ds, di, dr, dt_, d_nn_reg])
        else:
            derivatives = jnp.array([ds, di, dr, dt_])
        if len(u) > self.state_dimension:
            derivatives = jnp.concatenate([derivatives, nn_q[1:]])
        return derivatives


# LOSS AND GRAD FUNCTIONS
def _compute_loss(sol, data):
    infected, dead, recovered = data[:, 0], data[:, 1], data[:, 2]
    # loss is sum of MSE on Infected and Recovered+Dead
    loss1 = get_mse(
        data=jnp.log(infected),
        pred=jnp.log(jnp.absolute(sol.ys[:, 1] + sol.ys[:, 3])),
    )
    loss2 = get_mse(
        data=jnp.log(recovered + dead),
        pred=jnp.log(jnp.absolute(sol.ys[:, 2])),
    )
    # loss1 = jnp.mean((jnp.log(infected) - jnp.log(jnp.absolute(sol.ys[:, 1] + sol.ys[:, 3]))) ** 2)
    # loss2 = jnp.mean((jnp.log(recovered + dead) - jnp.log(jnp.absolute(sol.ys[:, 2]))) ** 2)
    return loss1 + loss2


@eqx.filter_jit
def compute_loss(
        prob_params, prob_static, u0, t_save, t_span, data, mlp_reg_exp, mlp_reg_strength, output_reg, adjoint=None
):
    prob = eqx.combine(prob_params, prob_static)
    if adjoint is not None:
        sol = prob.solve(diffrax.Tsit5(), u0, t_save, t_span, adjoint=adjoint)
    else:
        sol = prob.solve(diffrax.Tsit5(), u0, t_save, t_span)
    loss = _compute_loss(sol, data)
    if prob.func.regularise_network:
        mlp_reg = regularise_mlp(prob.func.mlp, mlp_reg_exp, mlp_reg_strength)
        mlp_output_reg = regularise_mlp_contribution(
            l2_norm_nn_contrib=sol.ys[:, 4],
            timespan_length=t_span[-1]-t_span[0],
            reg_strength=output_reg
        ) # TODO @GiacomoFabrini any way to avoid magic numbers?
        loss += (mlp_reg + mlp_output_reg)
        return loss, mlp_output_reg
    return loss, None


def compute_loss_grad(prob_params, prob_static, u0, t_save, t_span, data, mlp_reg_exp, mlp_reg_strength, output_reg):
    return eqx.filter_value_and_grad(compute_loss, has_aux=True)(
        prob_params, prob_static, u0, t_save, t_span, data, mlp_reg_exp, mlp_reg_strength, output_reg, adjoint=None
    )


@eqx.filter_jit
def compute_loss_optx(prob_params, args):
    prob_static, u0, t_save, t_span, data, mlp_reg_exp, mlp_reg_strength, output_reg = args
    loss, _ = compute_loss(
        prob_params, prob_static, u0, t_save, t_span, data, mlp_reg_exp, mlp_reg_strength, output_reg, None
    )
    return loss, None


# HESSIAN
# MATRIX -- NOT IN USE
# @eqx.filter_jit
# @jax.hessian
# def compute_hess(
#         prob_params, prob_static,
#         u0, t_save_train, t_span_train, data, adjoint,
#         reg_exp, reg_strength
# ):
#     prob = eqx.combine(prob_params, prob_static)
#     sol = prob.solve(diffrax.Tsit5(), u0, t_save_train, t_span_train, adjoint=adjoint)
#     return _compute_loss(sol, data) + regularise_mlp(prob.func.mlp, reg_exp, reg_strength)

# ARRAY FORM
@eqx.filter_jit
@jax.hessian
def compute_hess_array(
        flat_p, unflatten_p, prob_static,
        u0, t_save, t_span, data, adjoint,
        mlp_reg_exp, mlp_reg_strength, output_reg
):
    prob_params = unflatten_p(flat_p)
    loss, _ = compute_loss(
        prob_params, prob_static, u0, t_save, t_span, data, mlp_reg_exp, mlp_reg_strength, output_reg, adjoint
    )
    return loss


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
