import diffrax
import fire
import jax.random as jr
import jax.numpy as jnp

from dataclasses import replace
from jax import config
from src.common import (Conf, EarlyStoppingParams, model_save_dir,
                        state_dims, true_dynamic_parameters)
from src.model_training.training_setup import (broadcast_bounds, get_data_and_funcs_for_train,
                                               get_initial_conditions, setup_mlp)
from src.model_training.training_util import setup_curriculum_learning, sample_dynamical_params, get_sindy_formula
from src.ude_problem import UDEProblem
from training import train
from training_configuration import MECHANISTIC_FIT_EPOCHS, N_EPOCHS, MAX_ITER_OPTX_FINETUNE


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
        dynamical_model: sample_dynamical_params(
            true_params=true_dynamic_parameters[dynamical_model],
            deviation=conf.multistart_deviation,
            seed=seed,
            scale=conf.dynamic_params_scale,
            method=conf.dynamic_params_perturb_method,
            **broadcast_bounds(
                bounds={
                    "lb": conf.dynamic_params_lb,
                    "ub": conf.dynamic_params_ub
                },
                shape=true_dynamic_parameters[dynamical_model].shape
            )
        )
        for dynamical_model in true_dynamic_parameters.keys()
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
 loss_dict, option_args, plot_args) = get_data_and_funcs_for_train(
        conf, true_dynamic_parameters[conf.dynamical_model].shape, u0
    )
optx_config, options = option_args

# Setup curriculum learning (if enabled)
# TODO @GiacomoFabrini - this fails if N_EPOCHS = 0
curriculum_stage_dict = setup_curriculum_learning(
        data=train_args.data,
        t_save=train_args.t_save,
        n_epoch=N_EPOCHS,
        shooting_mode=conf.shooting_mode,
        shooting_window_length=conf.shooting_window_length,
        curriculum_learning=conf.curriculum_learning,
        num_curriculum_steps=conf.num_curriculum_steps,
        max_curriculum_epochs=conf.max_curriculum_epochs,
    )

ude_problem = UDEProblem(
    dynamical_model=conf.dynamical_model,
    dynamic_parameters=dynamic_parameters[conf.dynamical_model],
    mlp=mlp,
    options=options,
    state_dimension=state_dims[conf.dynamical_model],
    training_horizon=train_args.data.shape[0],
    shooting_mode=conf.shooting_mode,
    shooting_window_length=conf.shooting_window_length,
    shooting_iv_init=conf.shooting_iv_init,
    t_save_train=train_args.t_save if type(train_args.t_save) == diffrax.SaveAt \
        else diffrax.SaveAt(ts=train_args.t_save),
    u0_guess=train_args.u0,
    init_random_seed=conf.multistart_job,
    curriculum_learning=conf.curriculum_learning,
    curriculum_params=curriculum_stage_dict[1] if conf.curriculum_learning else None,
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
#                            TRAIN
# ================================================================
training_output = train(
    model=ude_problem,
    train_args=train_args,
    val_args=val_args,
    curriculum_params=curriculum_stage_dict if conf.curriculum_learning else None,
    conf=conf,
    early_stopping_params=early_stopping_params,
    loss_dict=loss_dict,
    optx_config=optx_config,
    max_iter_optx_mechanistic=MECHANISTIC_FIT_EPOCHS,
    n_epochs=N_EPOCHS,
    max_iter_optx_finetune=MAX_ITER_OPTX_FINETUNE,
    model_save_dir=model_save_dir,
    plot_args=plot_args,
)

# ================================================================
#                  GET SYMBOLIC EXPRESSION (SINDy)
# ================================================================
# SINDy on final CL model (if enabled)
if conf.serialise_curriculum_models:
    best_model, curriculum_models = training_output
    if conf.get_sindy_formula:
        # Final CL step (complete dataset fit) -> evaluate on validation dataset
        # TODO @GiacomoFabrini - do we want train or val?
        sindy_model = curriculum_models[list(curriculum_models.keys())[-1]]
        get_sindy_formula(
            model=sindy_model,
            u0=u0_val,
            t_save=t_save_validation,
            t_span=t_span_validation,
            settings=["MLP", "UDE"],
            use_derivatives=conf.sindy_use_derivatives,
        )
else:
    best_model = training_output
