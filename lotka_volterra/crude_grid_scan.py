import itertools as itt
import jax.random as jr
import jax.nn as jnn
import jax.numpy as jnp
import lotka_volterra.setup_diffrax_model as setup_lv

from common import Conf, EarlyStoppingParams, set_weight_bias
from example import build_MLP, get_training_validation_data
from training import train
from training_configuration import (DYNAMICAL_MODEL,
                                    MLP_WIDTHS,
                                    MLP_DEPTHS,
                                    MLP_ACTIVATIONS,
                                    OPTIMISERS,
                                    USE_EARLY_STOP,
                                    N_EPOCHS)

# Dictionary of activation functions to choose from
activation_functions = {
    "tanh": jnn.tanh,
    "relu": jnn.relu,
    "swish": jnn.swish,
    "elu": jnn.elu,
    "gelu": jnn.gelu,
}

# Key
key = jr.PRNGKey(123)
key_mlp, key_init = jr.split(key, 2)

# Get training and validation data
# TODO @GiacomoFabrini find better way to express filepath and pass it as argument
train_data, t_save_train, validation_data, t_save_validation = get_training_validation_data()


for dynamical_model, mlp_layer_width, mlp_depth, activation_fn_name, optimiser, use_early_stop in itt.product(
        DYNAMICAL_MODEL, MLP_WIDTHS, MLP_DEPTHS, MLP_ACTIVATIONS, OPTIMISERS, USE_EARLY_STOP
):
    conf = Conf(
        dynamical_model=dynamical_model,
        mlp_layer_width=mlp_layer_width,
        mlp_depth=mlp_depth,
        activation_fn_name=activation_fn_name,
        optimiser=optimiser,
        use_early_stop=use_early_stop
    )

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


    # Linear schedule Fabian set up for the DMM
    schedule_config = dict(
        init_value=conf.optimiser_learning_rate,
        transition_steps=N_EPOCHS/10,  # 100 when using 1000 epochs
        end_value=conf.optimiser_learning_rate/10,
    )

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
        conf=conf.__dict__,
        schedule_config=schedule_config,
        n_epochs=N_EPOCHS,
        early_stopping_params=early_stopping_params,
    )
