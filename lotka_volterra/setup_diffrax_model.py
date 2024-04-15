import jax.flatten_util
import jax.numpy as jnp
import diffrax
import equinox as eqx 
import jax
import jax.random as jr
import os
import pandas as pd

import lotka_volterra.common as common

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
        d_prey = α*prey + nn_out[0]
        d_predator = nn_out[1] - δ*predator
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

    def __init__(self, u0, dynamic_parameters, mlp, **kwargs):
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
              controller = diffrax.PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=1e-8, atol=1e-8)):
        t0, t1 = t_span
        sol = diffrax.diffeqsolve(diffrax.ODETerm(self.func),
                                  solver,
                                  t0, 
                                  t1, 
                                  None, 
                                  u0,
                                  saveat=t_save, 
                                  throw=False,
                                  stepsize_controller=controller)
        return sol

@eqx.filter_value_and_grad
def compute_loss_grad(prob, u0, t_save, t_span, data):
    u_pred = prob.solve(diffrax.Tsit5(), u0, t_save, t_span)
    return jnp.sum((u_pred.ys - data)**2)

@eqx.filter_jit
def compute_loss(prob, u0, t_save, t_span, data):
    u_pred = prob.solve(diffrax.Tsit5(), u0, t_save, t_span)
    return jnp.sum((u_pred.ys - data)**2)

# Setup the model (this mlp was used in the paper)
width_size = 5
hidden_layers = 1
key = jr.PRNGKey(123)
mlp = eqx.nn.MLP(in_size=2,
                 out_size=2,
                 width_size=width_size,
                 depth=hidden_layers,
                 activation=jax.nn.tanh,
                 key=key)
# For Neural ODEs (and what I have seen for UDEs) starting with small 
# weights and zero bias is helpful. Hence I wrote a helper function 
# for setting bias to zero, and use a glorot_uniform initialization with 
# a user chosen scale (default scale is 1)
# For the benchmark against Julia we can use import functions (in Common.py)
# to set equal weight, bias etc..., between the software, but first we should 
# focus on different training schemes :)
mlp = common.set_weight_bias(mlp, 0.1, key)

# Setting up the UDEProblem, I happen to use the true dynamic parameters 
# here, but this can be looked at later 
u0 = jnp.array([0.44249296, 4.6280594])
dynamic_parameters = jnp.array([1.3, 1.8])
uprob = UDEProblem(u0, dynamic_parameters, mlp)

# Parse training and validation data
train_data_df = pd.read_csv(os.path.join("Intermediate", "Lotka_volterra", "Train_data.csv"))
train_data = jnp.array(train_data_df[["u1", "u2"]].values)
train_t_save = diffrax.SaveAt(ts = train_data_df["t"].values)
validation_data_df = pd.read_csv(os.path.join("Intermediate", "Lotka_volterra", "Validation_data.csv"))
validation_data = jnp.array(validation_data_df[["u1", "u2"]].values)
validation_t_save = diffrax.SaveAt(ts = validation_data_df["t"].values)

# Compute loss and gradient
loss, grad = compute_loss_grad(uprob, u0, train_t_save, [0.0, 10.0], train_data)
# Compute validation loss
loss_validation = compute_loss(uprob, u0, validation_t_save, [0.0, 20.0], validation_data)


