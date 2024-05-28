import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import src.model_examples.lotka_volterra.setup_diffrax_model as setup_lv
import matplotlib.pyplot as plt
import os
import pandas as pd

from src.plotting import plot_fit
from src.model_training.model_util import set_weight_bias
from src.ude_problem import UDEProblem


# Flatten a tree to an array to allow gradient and Hessian functions to 
# produce Array output
def flatten(v):
    def f(v):
        leaves, _ = jax.tree_util.tree_flatten(v)
        return jnp.concatenate([x.ravel() for x in leaves])

    out, pullback = jax.vjp(f, v)
    return out, lambda x: pullback(x)[0]


# Set up the model (this mlp was used in the paper)
width_size = 5
hidden_layers = 2
# they say they have 2 hidden layers with 5 neurons, whereas this results in
# an input layer, a hidden layer with 5 neurons and an output layer
key = jr.PRNGKey(123)
mlp = eqx.nn.MLP(in_size=2,
                 out_size=2,
                 width_size=width_size,
                 depth=hidden_layers,
                 activation=jax.nn.tanh,
                 key=key)
# For Neural ODEs (and what I have seen for UDEs) starting with small 
# weights and zero bias is helpful. Hence, I wrote a helper function
# for setting bias to zero, and use a glorot_uniform initialization with 
# a user chosen scale (default scale is 1)
# For the benchmark against Julia we can use import functions (in Common.py)
# to set equal weight, bias etc..., between the software, but first we should 
# focus on different training schemes :)
mlp = set_weight_bias(mlp, 0.1, key)

# Setting up the UDEProblem, I happen to use the true dynamic parameters 
# here, but this can be looked at later 
u0 = jnp.array([0.44249296, 4.6280594])
dynamic_parameters = jnp.array([1.3, 1.8])
uprob = UDEProblem("lotka_volterra", dynamic_parameters, mlp)

# Read training and validation data
train_data_df = pd.read_csv(os.path.join("data_dir", "lotka_volterra", "Train_data.csv"))
train_data = jnp.array(train_data_df[["u1", "u2"]].values)
t_save_train = diffrax.SaveAt(ts=train_data_df["t"].values)
validation_data_df = pd.read_csv(os.path.join("data_dir", "lotka_volterra", "Validation_data.csv"))
validation_data = jnp.array(validation_data_df[["u1", "u2"]].values)
t_save_validation = diffrax.SaveAt(ts=validation_data_df["t"].values)

# Compute loss and gradient
loss, grad = setup_lv.compute_loss_grad(uprob, u0, t_save_train, [0.0, 10.0], train_data)

# We can easily plot the fit by first generating the data to plot, and then call plotnine (the data 
# should also be easy to plot with matplotlib :)). If you change below to t_save_validation and
# validation_data you get the plot for the validation. The solid line is the model, while the 
# dots are the data
df_pred_train, df_obs_train = setup_lv.get_result_plot(uprob, u0, t_save_train, [0.0, 10.0], train_data)
p = plot_fit(df_obs=df_obs_train, df_pred=df_pred_train)
# p
plt.show()

# Compute validation loss
loss_validation = setup_lv.compute_loss(uprob, u0, t_save_validation, [0.0, 20.0], validation_data)
df_pred_val, df_obs_val = setup_lv.get_result_plot(uprob, u0, t_save_validation, [0.0, 20.0], validation_data)


# Something fun, let us try the Hessian. Here we must use partition for not
# run into Jax problems. I have also created two Hessian functions, one that 
# outputs the Hessian as a Matrix, and one as a PyTree. This because a lot 
# of the nice abstractions with PyTree quickly become complicated when 
# dealing with the second derivative matrix
@eqx.filter_jit
@jax.hessian
def compute_hess(prob_params, prob_static, u0, t_save, t_span, data, adjoint):
    prob = eqx.combine(prob_params, prob_static)
    u_pred = prob.solve(diffrax.Tsit5(), u0, t_save, t_span,
                        adjoint=adjoint)
    return jnp.sum((u_pred.ys - data) ** 2)


@eqx.filter_jit
@jax.hessian
def compute_hess_array(flat_p, unflatten_p, prob_static, u0, t_save, t_span, data, adjoint):
    prob_params = unflatten_p(flat_p)
    prob = eqx.combine(prob_params, prob_static)
    u_pred = prob.solve(diffrax.Tsit5(), u0, t_save, t_span,
                        adjoint=adjoint)
    return jnp.sum((u_pred.ys - data) ** 2)


params, static = eqx.partition(uprob, eqx.is_array)
hess = compute_hess(params, static, u0, t_save_train, [0.0, 10.0], train_data,
                    diffrax.DirectAdjoint())

flat_p, unflatten_p = jax.flatten_util.ravel_pytree(params)
hess = compute_hess_array(flat_p, unflatten_p, static, u0, t_save_train, [0.0, 10.0], train_data,
                          diffrax.DirectAdjoint())
