import fire
import jax.random as jr
import jax.numpy as jnp
import src.model_examples.lotka_volterra.setup_diffrax_model as setup_lv
import src.model_examples.sir.setup_diffrax_model as setup_sir

from dataclasses import replace
from jax import config
from src.common import (activation_functions, Conf, EarlyStoppingParams, model_save_dir,
                        u0s, state_dims, true_dynamic_parameters)
from src.model_training.model_util import set_weight_bias
from src.model_training.training_util import sample_dynamical_params, get_sindy_formula, generate_epochs_dict
from src.model_training.landscape_plotting_util import analyse_lv_loss_landscape
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

# Copy default initial conditions
u0 = u0s.copy()

# Apply NODE-like augmentation
if conf.node_augmentation > 0:
    for dynamical_model in u0.keys():
        u0[dynamical_model] = jnp.concatenate([u0[dynamical_model], jnp.zeros(conf.node_augmentation)])

# Sample dynamic parameters and apply correct estimation scale (linear/log10)
if conf.training_setting != "NODE":
    # Sample dynamic parameters and convert to correct scale (linear/log10)
    dynamic_parameters = {
        dynamical_model: sample_dynamical_params(
            true_params=true_dynamic_parameters[dynamical_model],
            deviation=0.0, # true params first time
            seed=seed,
            scale=conf.dynamic_params_scale,
            method=conf.dynamic_params_perturb_method,
        )
        for dynamical_model in true_dynamic_parameters.keys()
    }
else:
    # if training NODEs, zero out dynamic parameters (no known terms), freeze them and set their scale to linear
    dynamic_parameters = {
        dynamical_model: jnp.zeros_like(true_dynamic_parameters[dynamical_model])
        for dynamical_model in true_dynamic_parameters.keys()
    }
    conf = replace(
        conf,
       dynamic_params_scale="linear",
       freeze_mechanistic_params_adam=True,
       freeze_mechanistic_params_optx=True
    )

# ================================================================
#                         SET UP MLP
# ================================================================
# Define input and output sizes for MLP
in_size = {
    'lotka_volterra': len(u0['lotka_volterra']),  # i.e. 2 (prey, predator) + conf.node_augmentation
    'sir': len(u0['sir'])-1,  # i.e. 4 (S, I, R) + conf.node_augmentation -- does not include t (quarantined)
}
out_size = {
    'lotka_volterra': 2+conf.node_augmentation,  # prey, predator, augmentation variables
    'sir': 1+conf.node_augmentation,  # Q (quarantine strength), augmentation variables
}

# Check activation function is valid
if conf.activation_fn_name not in activation_functions:
    raise ValueError(f"Unknown activation function: {conf.activation_fn_name}!")

# Build the MLP, initialise its weights and biases
mlp = set_weight_bias(
        build_mlp(
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
    ),
    conf.init_scale,
    key_init
)

# ================================================================
#                   SET UP UDE PROBLEM TO TRAIN
# ================================================================
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

# ================================================================
#                         TRAIN - MLP
# ================================================================
# Freeze mechanistic params -- set to ground truth values
print("Freezing mechanistic params...")
conf = replace(
    conf,
    curriculum_learning=True,
    freeze_mechanistic_params_adam=True,
    freeze_mechanistic_params_optx=True,
    freeze_nn_adam=False,
    freeze_nn_optx=False,
    serialise_curriculum_models=False
)
print("Training MLP...")
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
    early_stopping_params=early_stopping_params,
    loss_fn=loss_fn,
    loss_grad_fn=loss_grad_fn,
    loss_optx=loss_optx,
    optx_config=optx_config,
    max_iter_optx_mechanistic=0,
    n_epochs=N_EPOCHS,  # only train using ADAM
    max_iter_optx_finetune=0,
    model_save_dir=model_save_dir,
)

# Get SINDy formulas for whole UDE and trained MLP
get_sindy_formula(
    model=trained_model,
    u0=u0[conf.dynamical_model],
    t_save=t_save_train,
    t_span=t_span_train,
    settings=["MLP", "UDE"],
    use_derivatives=True,
)

# ================================================================
#                         TRAIN - dynamic params
# ================================================================
# Freeze mechanistic params -- set to ground truth values
print("Freezing MLP...")
conf = replace(
    conf,
    curriculum_learning=True,
    freeze_mechanistic_params_adam=False,
    freeze_mechanistic_params_optx=False,
    freeze_nn_adam=True,
    freeze_nn_optx=True,
    serialise_curriculum_models=True
)

# Sample with appropriate range
dynamic_parameters = {
    dynamical_model: sample_dynamical_params(
        true_params=true_dynamic_parameters[dynamical_model],
        deviation=conf.multistart_deviation,  # now sample the parameters
        seed=seed,
        scale=conf.dynamic_params_scale,
        method=conf.dynamic_params_perturb_method,
    )
    for dynamical_model in true_dynamic_parameters.keys()
}

# and replace within the trained UDE model -> random dynamic parameters, trained MLP
reinit_model = trained_model.copy(
    new_dynamic_parameters=dynamic_parameters[conf.dynamical_model],
)

initial_dynamic_params = reinit_model.func.dynamic_parameters
if conf.dynamic_params_scale == "log10":
    initial_dynamic_params = jnp.power(10, initial_dynamic_params)
print(f"Initial dynamic parameters: {initial_dynamic_params}")

# Re-train
print("Training dynamic parameters...")
final_model, curriculum_models = train(
    model=reinit_model,  # initialise with trained model with new dynamic params
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

# Get dynamic parameters at the beginning and end of each curriculum step
cl_epochs = generate_epochs_dict(sorted(list(curriculum_models.keys())))
opt_dynamic_params = {
    cl_step: {
        timepoint: curriculum_models[cl_epochs[cl_step][timepoint]].func.dynamic_parameters
        for timepoint in ["start", "end"]
    }
    for cl_step in range(conf.num_curriculum_steps)
}

final_dynamic_params = final_model.func.dynamic_parameters
if conf.dynamic_params_scale == "log10":
    final_dynamic_params = jnp.power(10, final_dynamic_params)
print(f"Final dynamic parameters: {final_dynamic_params}")
print(f"Ground truth parameters: {true_dynamic_parameters[conf.dynamical_model]}")

landscape_df = analyse_lv_loss_landscape(
    udeprob=curriculum_models[sorted(list(curriculum_models.keys()))[-1]],
    u0=jnp.concatenate([u0[conf.dynamical_model], jnp.zeros(1)]),  # augment initial condition for network regularisation
    train_data=train_data,
    t_save_train=t_save_train,
    t_span_train=t_span_train,
    validation_data=validation_data,
    t_save_validation=t_save_validation,
    t_span_validation=t_span_validation,
    conf=conf,
    N_EPOCHS=N_EPOCHS,
    opt_dynamic_params=opt_dynamic_params,
    num_points=200,
)
