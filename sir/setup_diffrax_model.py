import jax.flatten_util
import jax.numpy as jnp
import diffrax
import equinox as eqx 
import jax
import pandas as pd

import common as common

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
        β, γ, δ = self.dynamic_parameters
        S, I, R, T = u
        nn_t = self.mlp(jnp.array([S, I, R]))

        dS = - β*S*(I)/u0[0]
        dI = β*S*(I)/u0[0] - γ*I - nn_t[0]*I/u0[0]
        dR = γ*I + δ*T
        dT =  nn_t[0]*I/u0[0] - δ*T

        return jnp.array([dS, dI, dR, dT])
    
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
              controller = diffrax.PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=1e-8, atol=1e-8), 
              adjoint=diffrax.RecursiveCheckpointAdjoint()):
        t0, t1 = t_span
        sol = diffrax.diffeqsolve(diffrax.ODETerm(self.func),
                                  solver,
                                  t0, 
                                  t1, 
                                  None,
                                  u0, 
                                  args = u0,
                                  saveat=t_save, 
                                  throw=False,
                                  stepsize_controller=controller,
                                  adjoint=adjoint)
        return sol

@eqx.filter_value_and_grad
def compute_loss_grad(prob, u0, t_save, t_span, data):
    sol = prob.solve(diffrax.Tsit5(), u0, t_save, t_span)
    return _compute_loss(sol, data)

@eqx.filter_jit
def compute_loss(prob, u0, t_save, t_span, data):
    sol = prob.solve(diffrax.Tsit5(), u0, t_save, t_span)
    return _compute_loss(sol, data)

def _compute_loss(sol, data):
    infected = data[:, 0]
    dead = data[:, 1]
    recovered = data[:, 2]
    loss1 = jnp.sum( (jnp.log(infected) - jnp.log(jnp.absolute(sol.ys[:, 1] + sol.ys[:, 3])))**2 )
    loss2 = jnp.sum( (jnp.log(jnp.absolute(recovered + dead)) - jnp.log(jnp.absolute(sol.ys[:, 2])))**2 )
    return loss1 + loss2

def get_result_plot(prob, u0, t_save, t_span, data):
    '''
    Transform observed and model prediction data into pd.DataFrames that 
    easily can be plotted with plotnine
    '''
    # Model prediction
    sol = prob.solve(diffrax.Tsit5(), u0, t_save, t_span)
    df_pred1 = pd.DataFrame({"time" : sol.ts,
                             "value" : jnp.log(jnp.absolute(sol.ys[:, 1] + sol.ys[:, 3])),
                             "id" : "Infected",
                             "datasource" : "model"})
    df_pred2 = pd.DataFrame({"time" : sol.ts,
                             "value" : jnp.log(jnp.absolute(sol.ys[:, 2])),
                             "id" : "Recovered + dead",
                             "datasource" : "model"})
    df_predict = pd.concat([df_pred1, df_pred2], names=False)

    # Observed data
    infected, dead, recovered = data[:, 0], data[:, 1], data[:, 2]
    df_obs1 = pd.DataFrame({"time" : sol.ts,
                            "value" : jnp.log(infected),
                            "id" : "Infected",
                            "datasource" : "observed"})
    df_obs2 = pd.DataFrame({"time" : sol.ts,
                            "value" : jnp.log(jnp.absolute(recovered + dead)),
                            "id" : "Recovered + dead",
                            "datasource" : "observed"})
    df_obs = pd.concat([df_obs1, df_obs2], names=False)

    return df_predict, df_obs