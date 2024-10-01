import fire
import jax.random as jr
import jax.numpy as jnp
from jax import config
import src.model_examples.lotka_volterra.setup_diffrax_model as setup_lv

from src.common import activation_functions, Conf, EarlyStoppingParams
from src.mlp_experiment.mlp_training_lv import mlp_train
from src.model_training.model_util import set_weight_bias
from src.ude_problem import build_mlp
from src.model_training.training_configuration import N_EPOCHS


"""
This script is used to train a MLP on the Lotka-Volterra system. 
In particular, the MLP is trained to predict the interaction terms, unknown in the UDE formulation, 
from time-series of the two populations (prey or x1, predator or x2).
"""

# Set jax config
config.update("jax_enable_x64", True)

# Fetch CLI arguments (MLP structure, etc.)
conf = fire.Fire(Conf)

# Key for MLP weight and bias init
key = jr.PRNGKey(123)
key_mlp, key_init = jr.split(key, 2)

# Dynamical model params that regulate interaction terms (regression targets)
alpha, delta = jnp.array([1.3, 1.8])
beta, gamma = jnp.array([0.9, 0.8])

# Setup input (x1, x2) and output (beta*x1*x2, gamma*x1*x2) sizes
in_size = 2
out_size = 2

if conf.activation_fn_name not in activation_functions:
    raise ValueError(f"Unknown activation function: {conf.activation_fn_name}!")

mlp = build_mlp(
    in_size=in_size,
    out_size=out_size,
    width_size=conf.mlp_layer_width,
    hidden_layers=conf.mlp_depth,
    key=key_mlp,
    use_bias=conf.use_bias,
    use_final_bias=conf.use_final_bias,
    activation=activation_functions[conf.activation_fn_name],
    final_activation=activation_functions[conf.final_activation_fn_name],
    use_dropout=conf.use_dropout,
    input_dropout_rate=conf.input_dropout_rate,
    hidden_dropout_rate=conf.hidden_dropout_rate,
)

# Initialise weights and biases of MLP
mlp = set_weight_bias(mlp, conf.init_scale, key_init)

# Get training and validation data
(train_data, t_save_train, t_span_train,
 validation_data, t_save_validation, t_span_validation) = setup_lv.get_training_validation_data()


targets = {}
for dataset, data in zip(["train", "val"], [train_data, validation_data]):
    prey = data[:, 0]
    predator = data[:, 1]
    if conf.mlpexp_target == "RHS":
        targets[dataset] = jnp.stack(
        [
                alpha * prey - beta * prey * predator,
                gamma * prey * predator - delta * predator
            ], axis=1
        )
    elif conf.mlpexp_target == "interaction":
        targets[dataset] = jnp.stack(
            [
                - beta * prey * predator,
                gamma * prey * predator
            ], axis=1
        )
    else:
        raise ValueError(f"Unknown mode: {conf.mlpexp_target}!")

# Setup early stopping
early_stopping_params = EarlyStoppingParams(
    use_early_stopping=conf.use_early_stop,  # enables flax.training.early_stopping
    patience=50,  # number of consecutive epochs where we tolerate loss_val not improving by at least min_improvement
    min_improvement=0,  # absolute improvement in loss_val to consider as improvement
)

trained_mlp = mlp_train(
    model=mlp,
    training_data=train_data,
    training_targets=targets["train"],
    validation_data=validation_data,
    validation_targets=targets["val"],
    conf=conf,
    n_epochs=N_EPOCHS,
    early_stopping_params=early_stopping_params,
)
