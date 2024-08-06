import jax.flatten_util
import jax.random as jr
import os
import pandas as pd
import diffrax
import equinox as eqx
import jax.numpy as jnp
import jax
import timeit
import matplotlib.pyplot as plt
import src.model_examples.sir.setup_diffrax_model as setup_sir

from src.model_training.model_util import set_weight_bias
from src.plotting import plot_fit
from src.ude_problem import UDEProblem

# Set up the model (this mlp was used in the paper)
width_size = 10
hidden_layers = 1
key = jr.PRNGKey(123)
mlp = eqx.nn.MLP(in_size=3,
                 out_size=1,
                 width_size=width_size,
                 depth=hidden_layers,
                 activation=jax.nn.relu,
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
u0 = jnp.array([51000000.0, 578, 24, 10])
dynamic_parameters = jnp.array([0.15, 0.013, 0.01])
options = {
    # "add_interaction": False,
    "dynamic_params_scale": False,
}
u_prob = UDEProblem("sir", dynamic_parameters, mlp, options)

# Read training and validation data
# Training
train_data_df = pd.read_csv(os.path.join("data_dir", "sir", "Korea_data_train.csv"))
train_data = jnp.array(train_data_df[["infected", "dead", "recovered"]].values)
t_save_train = diffrax.SaveAt(ts=train_data_df["time"].values)
t_span_train = [0.0, max(train_data_df["time"])]
# Validation
validation_data_df = pd.read_csv(os.path.join("data_dir", "sir", "Korea_data_validation.csv"))
validation_data = jnp.array(validation_data_df[["infected", "dead", "recovered"]].values)
t_save_validation = diffrax.SaveAt(ts=validation_data_df["time"].values)
t_span_validation = [0.0, max(validation_data_df["time"])]

# Compute loss and gradient
loss = setup_sir.compute_loss(u_prob, u0, t_save_train, t_span_train, train_data)
loss, grad = setup_sir.compute_loss_grad(u_prob, u0, t_save_train, t_span_train, train_data)
# Compute loss for the validation data
loss_validation = setup_sir.compute_loss(u_prob, u0, t_save_validation, t_span_validation, validation_data)

# Just as for the Lotka-Volterra model we can easily plot the results. Note, the 
# observables are on log-scale for this model
# Training data
df_pred_train, df_obs_train = setup_sir.get_result_plot(u_prob, u0, t_save_train, t_span_train, train_data)
p = plot_fit(df_obs=df_obs_train, df_pred=df_pred_train)
# p
plt.show()
# Validation data
df_pred_val, df_obs_val = setup_sir.get_result_plot(u_prob, u0, t_save_validation, t_span_validation, validation_data)
p = plot_fit(df_obs=df_obs_val, df_pred=df_pred_val)
# p
plt.show()

# Something fun, let us try the Hessian. Here we must use partition for not
# run into Jax problems. I have also created two Hessian functions, one that 
# outputs the Hessian as a Matrix, and one as a PyTree. This because a lot 
# of the nice abstractions with PyTree quickly become complicated when 
# dealing with the second derivative matrix


params, static = eqx.partition(u_prob, eqx.is_array)
hess = setup_sir.compute_hess(params, static, u0, t_save_train, t_span_train, train_data,
                              diffrax.DirectAdjoint())

flat_p, unflatten_p = jax.flatten_util.ravel_pytree(params)
hess = setup_sir.compute_hess_array(flat_p, unflatten_p, static, u0, t_save_train, t_span_train, train_data,
                                    diffrax.DirectAdjoint())


# We can run some fun timing
def time_stuff():
    def _grad():
        loss, grad = setup_sir.compute_loss_grad(u_prob, u0, t_save_train, t_span_train, train_data)

    def _hess():
        hess = setup_sir.compute_hess_array(flat_p, unflatten_p, static, u0, t_save_train, t_span_train,
                                            train_data, diffrax.DirectAdjoint())

    duration1 = timeit.Timer(_grad).repeat(100, number=1)
    duration2 = timeit.Timer(_hess).repeat(100, number=1)
    return min(duration1), min(duration2)


b_grad, b_hess = time_stuff()
# Grad ~ 12ms
# Hess ~ 48ms
