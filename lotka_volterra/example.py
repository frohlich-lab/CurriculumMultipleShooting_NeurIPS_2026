import jax.random as jr
import os
import pandas as pd
import diffrax
import equinox as eqx
import jax.numpy as jnp
import jax

import lotka_volterra.common as common
import lotka_volterra.setup_diffrax_model as setup_lv

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
uprob = setup_lv.UDEProblem(u0, dynamic_parameters, mlp)

# Parse training and validation data
train_data_df = pd.read_csv(os.path.join("Intermediate", "Lotka_volterra", "Train_data.csv"))
train_data = jnp.array(train_data_df[["u1", "u2"]].values)
train_t_save = diffrax.SaveAt(ts = train_data_df["t"].values)
validation_data_df = pd.read_csv(os.path.join("Intermediate", "Lotka_volterra", "Validation_data.csv"))
validation_data = jnp.array(validation_data_df[["u1", "u2"]].values)
validation_t_save = diffrax.SaveAt(ts = validation_data_df["t"].values)

# Compute loss and gradient
loss, grad = setup_lv.compute_loss_grad(uprob, u0, train_t_save, [0.0, 10.0], train_data)
# Compute validation loss
loss_validation = setup_lv.compute_loss(uprob, u0, validation_t_save, [0.0, 20.0], validation_data)