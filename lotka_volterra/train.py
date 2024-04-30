import fire
import jax.random as jr
import jax.nn as jnn
import jax.numpy as jnp
import lotka_volterra.setup_diffrax_model as setup_lv

from common import build_MLP, Conf, EarlyStoppingParams, set_weight_bias
from training import train
# from training_configuration import DYNAMICAL_MODEL, MLP_WIDTH, MLP_DEPTH

conf = fire.Fire(Conf)

# Dictionary of activation functions to choose from
activation_functions = {
    "tanh": jnn.tanh,
    "relu": jnn.relu,
    "swish": jnn.swish,
    "elu": jnn.elu,
    "gelu": jnn.gelu,
}

# Get training and validation data
# TODO @GiacomoFabrini find better way to express filepath and pass it as argument
train_data, t_save_train, validation_data, t_save_validation = setup_lv.get_training_validation_data_lotka_volterra()
# Key
key = jr.PRNGKey(123)
key_mlp, key_init = jr.split(key, 2)
# Setup MLP (same as the original paper for now)
mlp = build_MLP(
    in_size=2,  # model state (prey, predator)
    out_size=2,  # model state (prey, predator)
    width_size=conf.mlp_layer_width,
    hidden_layers=conf.mlp_depth,
    key=key_mlp,
    activation=activation_functions[conf.activation_fn_name],
)
# Initialise weights and biases of MLP
mlp = set_weight_bias(mlp, conf.init_scale, key_init)
# Get u0
u0 = jnp.array([0.44249296, 4.6280594])
# Get dynamic parameters (for now true ones are used)
dynamic_parameters = jnp.array([1.3, 1.8])
# Setup UDEProblem to train
ude_problem = setup_lv.UDEProblem(dynamic_parameters, mlp)


# # Linear schedule Fabian set up for the DMM -- DISABLED WITH NEW SCHEDULER
# schedule_config = dict(
#     init_value=conf.optimiser_learning_rate,
#     transition_steps=100,
#     end_value=conf.optimiser_learning_rate/10,
# )

early_stopping_params = EarlyStoppingParams(
    use_early_stopping=conf.use_early_stop,  # enables flax.training.early_stopping
    patience=50,  # number of consecutive epochs where we tolerate loss_val not improving by at least min_improvement
    min_improvement=0,  # absolute improvement in loss_val to consider as improvement
)

trained_model = train(
    model=ude_problem,
    u0=u0,
    training_data=train_data,
    t_save_train=t_save_train,
    t_span_train=[0.0, 10.0],
    validation_data=validation_data,
    t_save_val=t_save_validation,
    t_span_val=[0.0, 20.0],
    conf=conf,
    n_epochs=5000,
    early_stopping_params=early_stopping_params,
)
