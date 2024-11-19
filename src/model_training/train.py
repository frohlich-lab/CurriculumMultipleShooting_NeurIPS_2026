import fire
import jax.random as jr
import jax.numpy as jnp
import src.model_examples.lotka_volterra.setup_diffrax_model as setup_lv
import src.model_examples.sir.setup_diffrax_model as setup_sir

from jax import config
from src.common import activation_functions, Conf, EarlyStoppingParams, model_save_dir
from src.model_training.model_util import set_weight_bias
from src.model_training.training_util import sample_dynamical_params
from src.ude_problem import UDEProblem, build_mlp
from training import train
from training_configuration import MECHANISTIC_FIT_EPOCHS, N_EPOCHS, MAX_ITER_OPTX_FINETUNE

conf = fire.Fire(Conf)

# Set seed for sampling dynamical parameters
seed = conf.multistart_job

# Technically Float32 should work, but for consistency between the benchmarks
# I think we should use Float64
config.update("jax_enable_x64", True)

# Key
key = jr.PRNGKey(seed)
key_mlp, key_init = jr.split(key, 2)

state_dims = {
    "lotka_volterra": 2,
    "sir": 4,
}

# Setup default u0, dynamic parameters, in_size, out_size
u0 = {
    'lotka_volterra': jnp.array([0.44249296, 4.6280594]),
    'sir': jnp.array([51000000.0, 578, 24, 10]),
}

if conf.node_augmentation > 0:
    for dynamical_model in u0.keys():
        u0[dynamical_model] = jnp.concatenate([u0[dynamical_model], jnp.zeros(conf.node_augmentation)])

true_dynamic_parameters = {
    'lotka_volterra': jnp.array([1.3, 1.8]),
    'sir': jnp.array([0.15, 0.013, 0.01]),
}

# Sample dynamic parameters and convert to correct scale (linear/log10)
dynamic_parameters = {
    dynamical_model: sample_dynamical_params(
        true_params=true_dynamic_parameters[dynamical_model],
        deviation=conf.multistart_deviation,
        seed=seed,
        scale=conf.dynamic_params_scale,
        method=conf.dynamic_params_perturb_method,
    )
    for dynamical_model in true_dynamic_parameters.keys()
}

in_size = {
    'lotka_volterra': len(u0['lotka_volterra']),  # i.e. 2 (prey, predator) + conf.node_augmentation
    'sir': len(u0['sir'])-1,  # i.e. 4 (S, I, R) + conf.node_augmentation -- does not include t (quarantined)
}

out_size = {
    'lotka_volterra': 2+conf.node_augmentation,  # prey, predator, augmentation variables
    'sir': 1+conf.node_augmentation,  # Q (quarantine strength), augmentation variables
}

# Setup MLP (same as the original paper for now)
if conf.activation_fn_name not in activation_functions:
    raise ValueError(f"Unknown activation function: {conf.activation_fn_name}!")

mlp = build_mlp(
    in_size=in_size[conf.dynamical_model],
    out_size=out_size[conf.dynamical_model],
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

# Setup UDEProblem to train
if conf.dynamical_model == 'lotka_volterra':
    # TODO @GiacomoFabrini find better way to express filepath and pass it as argument
    (train_data, t_save_train, t_span_train,
     validation_data, t_save_validation, t_span_validation) = setup_lv.get_training_validation_data()

    loss_fn = setup_lv.compute_loss
    loss_grad_fn = setup_lv.compute_loss_grad
    loss_optx = setup_lv.compute_loss_optx
    optx_config = {
        "rtol": 1e-6,  # changed from 1e-3
        "atol": 1e-6,  # changed from 1e-3
    }
elif conf.dynamical_model == 'sir':
    # TODO @GiacomoFabrini find better way to express filepath and pass it as argument - same as above
    (train_data, t_save_train, t_span_train,
     validation_data, t_save_validation, t_span_validation) = setup_sir.get_training_validation_data()

    loss_fn = setup_sir.compute_loss
    loss_grad_fn = setup_sir.compute_loss_grad
    loss_optx = setup_sir.compute_loss_optx
    optx_config = {
         "rtol": 1e-6,  # changed from 1e-4
         "atol": 1e-6,  # changed from 1e-4
    }
else:
    raise ValueError(f"Unknown dynamical model: {conf.dynamical_model}!")

options = {
    # "add_interaction": conf.add_interaction,
    "dynamic_params_scale": conf.dynamic_params_scale,
}
ude_problem = UDEProblem(
    dynamical_model=conf.dynamical_model,
    dynamic_parameters=dynamic_parameters[conf.dynamical_model],
    mlp=mlp,
    options=options,
    state_dimension=state_dims[conf.dynamical_model],
)

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

# TODO @GiacomoFabrini - can we get u0 from data? fix this!
# u0 = train_data[0]
# if conf.node_augmentation > 0:
#     u0 = jnp.concatenate([u0, jnp.zeros(conf.node_augmentation)])

training_output = train(
    model=ude_problem,
    u0=u0[conf.dynamical_model],
    training_data=train_data,
    t_save_train=t_save_train,
    t_span_train=t_span_train,
    validation_data=validation_data,
    t_save_val=t_save_validation,
    t_span_val=t_span_validation,
    conf=conf,
    early_stopping_params=early_stopping_params,
    loss_fn=loss_fn,
    loss_grad_fn=loss_grad_fn,
    loss_optx=loss_optx,
    optx_config=optx_config,
    max_iter_optx_mechanistic=MECHANISTIC_FIT_EPOCHS,
    n_epochs=N_EPOCHS,
    max_iter_optx_finetune=MAX_ITER_OPTX_FINETUNE,
    model_save_dir=model_save_dir,
)
