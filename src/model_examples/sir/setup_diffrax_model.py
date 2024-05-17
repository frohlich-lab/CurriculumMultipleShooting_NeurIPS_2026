import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import os
import pandas as pd

from src.common import plot_fit, regularise_mlp
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
        u0 = args
        beta, gamma, delta = self.dynamic_parameters
        s, i, r, t = u
        nn_t = self.mlp(jnp.array([s, i, r]))

        ds = - beta * s * i / u0[0]
        di = beta * s * i / u0[0] - gamma * i - nn_t[0] * i / u0[0]
        dr = gamma * i + delta * t
        dt = nn_t[0] * i / u0[0] - delta * t

        return jnp.array([ds, di, dr, dt])


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
                                  args=u0,
                                  saveat=t_save,
                                  throw=False,
                                  stepsize_controller=controller,
                                  adjoint=adjoint)
        return sol


@eqx.filter_value_and_grad
def compute_loss_grad(prob, u0, t_save, t_span, data, reg_exp, reg_strength):
    sol = prob.solve(diffrax.Tsit5(), u0, t_save, t_span)
    return _compute_loss(sol, data) + regularise_mlp(prob.func.mlp, reg_exp, reg_strength)


@eqx.filter_jit
def compute_loss(prob, u0, t_save, t_span, data, reg_exp, reg_strength):
    sol = prob.solve(diffrax.Tsit5(), u0, t_save, t_span)
    return _compute_loss(sol, data) + regularise_mlp(prob.func.mlp, reg_exp, reg_strength)


@eqx.filter_jit
def compute_loss_optx(prob_params, args):
    prob_static, u0, t_save, t_span, data = args
    prob = eqx.combine(prob_params, prob_static)
    sol = prob.solve(diffrax.Tsit5(), u0, t_save, t_span)
    return _compute_loss(sol, data)


def _compute_loss(sol, data):
    infected = data[:, 0]
    dead = data[:, 1]
    recovered = data[:, 2]
    loss1 = jnp.mean((jnp.log(infected) - jnp.log(jnp.absolute(sol.ys[:, 1] + sol.ys[:, 3]))) ** 2)
    loss2 = jnp.mean((jnp.log(jnp.absolute(recovered + dead)) - jnp.log(jnp.absolute(sol.ys[:, 2]))) ** 2)
    return loss1 + loss2


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


def save_plot_results(
        trained_model: UDEProblem,
        u0: jnp.array,
        t_save: diffrax.SaveAt,
        t_span: List,
        data: jnp.array,
        dataset: str,
        plot_name: str,
        plot_fmt: str,
        return_plot: bool = False,
):
    df_obs, df_pred = get_result_plot(
        trained_model,
        u0,
        t_save,
        t_span,
        data,
    )
    p = plot_fit(df_obs, df_pred)
    if return_plot:
        return p
    else:
        p.save(f"{dataset}_{plot_name}.{plot_fmt}")


def get_training_validation_data():
    # Read training and validation data
    # Training
    train_data_df = pd.read_csv(os.path.join("src", "data_dir", "SIR", "Korea_data_train.csv"))
    train_data = jnp.array(train_data_df[["infected", "dead", "recovered"]].values)
    t_save_train = diffrax.SaveAt(ts=train_data_df["time"].values)
    t_span_train = [0.0, max(train_data_df["time"])]
    # Validation
    validation_data_df = pd.read_csv(os.path.join("src", "data_dir", "SIR", "Korea_data_validation.csv"))
    validation_data = jnp.array(validation_data_df[["infected", "dead", "recovered"]].values)
    t_save_validation = diffrax.SaveAt(ts=validation_data_df["time"].values)
    t_span_validation = [0.0, max(validation_data_df["time"])]
    return train_data, t_save_train, t_span_train, validation_data, t_save_validation, t_span_validation
