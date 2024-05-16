import fire
import jax.random as jr
import jax.nn as jnn
import jax.numpy as jnp
import lotka_volterra.setup_diffrax_model as setup_lv
import sir.setup_diffrax_model as setup_sir

from common import build_mlp, Conf, EarlyStoppingParams, set_weight_bias
from training import train
from training_configuration import N_EPOCHS, MAX_ITER_OPTX

conf = fire.Fire(Conf)

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

# Setup default u0, dynamic parameters, in_size, out_size
u0 = {
    'lotka_volterra': jnp.array([0.44249296, 4.6280594]),
    'SIR': jnp.array([51000000.0, 578, 24, 10]),
}

dynamic_parameters = {
    'lotka_volterra': jnp.array([1.3, 1.8]),
    'SIR': jnp.array([0.15, 0.013, 0.01]),
}

in_size = {
    'lotka_volterra': 2,
    'SIR': 3,
}

out_size = {
    'lotka_volterra': 2,
    'SIR': 1,
}

# Setup MLP (same as the original paper for now)
mlp = build_mlp(
    in_size=in_size[conf.dynamical_model],
    out_size=out_size[conf.dynamical_model],
    width_size=conf.mlp_layer_width,
    hidden_layers=conf.mlp_depth,
    key=key_mlp,
    activation=activation_functions[conf.activation_fn_name],
)
# Initialise weights and biases of MLP
mlp = set_weight_bias(mlp, conf.init_scale, key_init)

# Setup UDEProblem to train
if conf.dynamical_model == 'lotka_volterra':
    # TODO @GiacomoFabrini find better way to express filepath and pass it as argument
    (train_data, t_save_train, t_span_train,
     validation_data, t_save_validation, t_span_validation) = setup_lv.get_training_validation_data()

    ude_problem = setup_lv.UDEProblem(dynamic_parameters[conf.dynamical_model], mlp)
    loss_fn = setup_lv.compute_loss
    loss_grad_fn = setup_lv.compute_loss_grad
    loss_optx = setup_lv.compute_loss_optx
    optx_config = {
        "rtol": 1e-3,
        "atol": 1e-3,
    }
elif conf.dynamical_model == 'SIR':
    # TODO @GiacomoFabrini find better way to express filepath and pass it as argument - same as above
    (train_data, t_save_train, t_span_train,
     validation_data, t_save_validation, t_span_validation) = setup_sir.get_training_validation_data()

    ude_problem = setup_sir.UDEProblem(dynamic_parameters[conf.dynamical_model], mlp)
    loss_fn = setup_sir.compute_loss
    loss_grad_fn = setup_sir.compute_loss_grad
    loss_optx = setup_sir.compute_loss_optx
    optx_config = {
        "rtol": 1e-4,
        "atol": 1e-4,
    }
else:
    raise ValueError(f"Unknown dynamical model: {conf.dynamical_model}!")

# # Linear schedule Fabian set up for the DMM -- DISABLED WITH NEW SCHEDULER
# schedule_config = dict(
#     init_value=conf.optimiser_learning_rate,
#     transition_steps=100,
#     end_value=conf.optimiser_learning_rate/10
# )

early_stopping_params = EarlyStoppingParams(
    use_early_stopping=conf.use_early_stop,  # enables flax.training.early_stopping
    patience=50,  # number of consecutive epochs where we tolerate loss_val not improving by at least min_improvement
    min_improvement=0,  # absolute improvement in loss_val to consider as improvement
)

trained_model = train(
    model=ude_problem,
    u0=u0[conf.dynamical_model],
    training_data=train_data,
    t_save_train=t_save_train,
    t_span_train=t_span_train,
    validation_data=validation_data,
    t_save_val=t_save_validation,
    t_span_val=t_span_validation,
    conf=conf,
    n_epochs=N_EPOCHS,
    early_stopping_params=early_stopping_params,
    loss_fn=loss_fn,
    loss_grad_fn=loss_grad_fn,
    loss_optx=loss_optx,
    optx_config=optx_config,
    max_iter_optx=MAX_ITER_OPTX,
)
