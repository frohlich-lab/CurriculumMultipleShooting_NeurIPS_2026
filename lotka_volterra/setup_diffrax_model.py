import jax.numpy as jnp
import diffrax
import equinox as eqx
import jax
import os
import pandas as pd

from common import plot_fit
from typing import List

# Technically Float32 should work, but for consistency between the benchmarks
# I think we should use Float64
jax.config.update("jax_enable_x64", True)


class UDEFunction(eqx.Module):
    """
    Class to evaluate right hand side (vector field) of an UDEModel

    Methods
    -------
    __call__(t, u, args)
        Evaluates the right hand side for input vector u, at time t
    """

    dynamic_parameters: jax.Array
    mlp: eqx.nn.MLP

    def __init__(self, dynamic_parameters, mlp, **kwargs):
        super().__init__(**kwargs)
        self.dynamic_parameters = dynamic_parameters
        self.mlp = mlp

    def __call__(self, t, u, args):
        α, δ = self.dynamic_parameters
        prey, predator = u
        nn_out = self.mlp(u)
        d_prey = α * prey + nn_out[0]
        d_predator = nn_out[1] - δ * predator
        return jnp.array([d_prey, d_predator])


class UDEProblem(eqx.Module):
    """
    Class to solve the UDEProblem with a chosen ODE-solver

    Methods
    -------
    solve(self, solver t_save, t_span)
        Solve the UDE for t_span = [t0, tend] and save the solution at the
        points in t_save using the provided solver. Controller is the adaptive
        ODE-stepping controller (should not need to be changed)
    """

    func: UDEFunction

    def __init__(self, dynamic_parameters, mlp, **kwargs):
        """
        Parameters
        ----------
        solver
            A valid diffrax ODE-solver (e.g. Tsit5 or Kvaerno5)
        dynamic_parameters : jax.Array
            Starting values for the model dynamic parameters (α, δ in the model
            equations).
        u0 : jax.Array
            Initial values for the UDEProblem. Technically we can estimate these,
            but at least for the Lotka-Volterra model this is not the case.
        mlp : eqx.nn.MLP
            An equinox Standard Multi-Layer Perceptron (MLP) network. If set to
            None path_meta must be provided.
        controller : diffrax.PIDController
        """
        super().__init__(**kwargs)
        self.func = UDEFunction(dynamic_parameters, mlp)

    def solve(self, solver, u0, t_save, t_span,
              controller=diffrax.PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=1e-8, atol=1e-8),
              adjoint=diffrax.RecursiveCheckpointAdjoint()):
        t0, t1 = t_span
        sol = diffrax.diffeqsolve(diffrax.ODETerm(self.func),
                                  solver,
                                  t0,
                                  t1,
                                  None,
                                  u0,
                                  saveat=t_save,
                                  throw=False,
                                  stepsize_controller=controller,
                                  adjoint=adjoint)
        return sol


@eqx.filter_value_and_grad
def compute_loss_grad(prob, u0, t_save, t_span, data, reg_exp, reg_strength):
    u_pred = prob.solve(diffrax.Tsit5(), u0, t_save, t_span)
    reg_loss = 0
    for layer_num in range(len(prob.func.mlp.layers)):
        w = prob.func.mlp.layers[layer_num].weight
        reg_loss += jnp.mean(
            jnp.abs(w)**reg_exp
        )
    return jnp.sum((u_pred.ys - data) ** 2) + reg_strength * reg_loss


@eqx.filter_jit
def compute_loss(prob, u0, t_save, t_span, data, reg_exp, reg_strength):
    u_pred = prob.solve(diffrax.Tsit5(), u0, t_save, t_span)
    reg_loss = 0
    for layer_num in range(len(prob.func.mlp.layers)):
        w = prob.func.mlp.layers[layer_num].weight
        reg_loss += jnp.mean(
            jnp.abs(w) ** reg_exp
        )
    return jnp.sum((u_pred.ys - data) ** 2) + reg_strength * reg_loss


@eqx.filter_jit
@jax.hessian
def compute_hess(prob_params, prob_static, u0, t_save, t_span, data, adjoint):
    prob = eqx.combine(prob_params, prob_static)
    u_pred = prob.solve(diffrax.Tsit5(), u0, t_save, t_span,
                        adjoint=adjoint)
    return jnp.mean((u_pred.ys - data)**2)


@eqx.filter_jit
@jax.hessian
def compute_hess_array(flat_p, unflatten_p, prob_static, u0, t_save, t_span, data, adjoint):
    prob_params = unflatten_p(flat_p)
    prob = eqx.combine(prob_params, prob_static)
    u_pred = prob.solve(diffrax.Tsit5(), u0, t_save, t_span,
                        adjoint=adjoint)
    return jnp.mean((u_pred.ys - data)**2)


def get_training_validation_data():
    # Read training and validation data
    train_data_df = pd.read_csv(os.path.join("Intermediate", "Lotka_volterra", "Train_data.csv"))
    train_data = jnp.array(train_data_df[["u1", "u2"]].values)
    t_save_train = diffrax.SaveAt(ts=train_data_df["t"].values)
    t_span_train = [0.0, max(train_data_df["t"])]
    validation_data_df = pd.read_csv(os.path.join("Intermediate", "Lotka_volterra", "Validation_data.csv"))
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


def save_plot_results(
        trained_model: UDEProblem,
        u0: jnp.array,
        t_save: diffrax.SaveAt,
        t_span: List,
        data: jnp.array,
        dataset: str,
        plot_name: str,
        plot_fmt: str,
):
    df_obs, df_pred = get_result_plot(
        trained_model,
        u0,
        t_save,
        t_span,
        data,
    )
    p = plot_fit(df_obs, df_pred)
    p.save(f"{dataset}_{plot_name}.{plot_fmt}")


# Idea: evaluate whether the output of the NN component learns the shape of prey*predator term (linear in both)
# STILL TO BE TESTED!
# @eqx.filter_jit
# def compute_corr(prob, data):
#     nn_out = prob.func.mlp(data.T)  # shape mismatch - how do I overcome this?
#     return jnp.corrcoef(nn_out, data)
