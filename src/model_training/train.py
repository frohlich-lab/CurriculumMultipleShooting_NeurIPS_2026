import fire
import jax.random as jr
import jax.numpy as jnp

from dataclasses import replace
from jax import config
from src.common import (
    Conf,
    data_dir,
    model_save_dir,
    state_dims,
    true_dynamic_parameters
)
from src.model_training.training_setup import (
    get_data_and_funcs_for_train,
    get_initial_conditions,
    load_multistart_dynamic_params,
    setup_mlp,
    setup_time_masks_u0_curriculum
)
from src.ude_problem import UDEProblem
from training import train


# ================================================================
#            GET CONFIG, SETUP PRECISION, GET PRNG KEYS
# ================================================================
conf = fire.Fire(Conf)

# Set seed for sampling dynamical parameters
seed = conf.multistart_job

# Configure JAX to use 64-bit precision
config.update("jax_enable_x64", True)

# Key
key = jr.PRNGKey(seed)
key_mlp, key_init = jr.split(key, 2)

# ================================================================
#           INITIAL CONDITIONS AND DYNAMIC PARAMETERS
# ================================================================
# Get initial conditions
u0 = get_initial_conditions(conf)

# Sample dynamic parameters and apply correct estimation scale
if conf.training_setting != "NODE":
    dynamic_parameters = {
        conf.dynamical_model: load_multistart_dynamic_params(conf, data_dir)
    }
else:
    # if training NODEs, zero out dynamic parameters (no known terms),
    # freeze them and set scale to linear
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
mlp = setup_mlp(conf, u0, key_mlp, key_init)

# ================================================================
#                   SET UP UDE PROBLEM TO TRAIN
# ================================================================
# Get data and functions for training
(train_args, val_args,
 loss_dict, option_args, plot_args, rescaling_args) = get_data_and_funcs_for_train(
        conf, true_dynamic_parameters[conf.dynamical_model].shape, u0
    )
optx_config, options = option_args

# Set up SaveAt windows, binary masks and initial values
t_save_windows, masks, initial_values, curriculum_stage_dict = setup_time_masks_u0_curriculum(
    conf=conf,
    train_args=train_args,
    u0_guess=train_args.u0,
    effective_state_dimension=state_dims[conf.dynamical_model] + conf.node_augmentation,
)
# Update plot args
plot_args["t_save_windows"] = t_save_windows

ude_problem = UDEProblem(
    dynamical_model=conf.dynamical_model,
    dynamic_parameters=dynamic_parameters[conf.dynamical_model],
    mlp=mlp,
    options=options,
    state_dimension=state_dims[conf.dynamical_model],
    initial_values=initial_values,
    regularise_network=True,  # TODO: consider making this more general / not hardcoded
)

# ================================================================
#                            TRAIN
# ================================================================
model_save_subdir = model_save_dir / conf.dynamical_model
model_save_subdir.mkdir(parents=True, exist_ok=True)

best_model = train(
    model=ude_problem,
    train_args=train_args,
    val_args=val_args,
    t_save_windows=t_save_windows,
    masks=masks,
    curriculum_params=curriculum_stage_dict if conf.curriculum_learning else None,
    conf=conf,
    loss_dict=loss_dict,
    optx_config=optx_config,
    n_epochs=conf.num_epochs_adam,
    max_iter_optx_finetune=conf.num_epochs_bfgs,
    model_save_dir=model_save_subdir,
    plot_args=plot_args,
    rescaling_args=rescaling_args,
)
