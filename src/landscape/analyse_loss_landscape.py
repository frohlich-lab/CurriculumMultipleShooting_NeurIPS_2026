import fire
# import jax.random as jr

from functools import partial
from jax import config
from src.common import Conf, model_save_dir, true_dynamic_parameters, max_iters_diffrax
from src.landscape.landscape_utils import *
from src.model_training.losses import compute_hess_array
from src.model_training.training_setup import (
    get_data_and_funcs_for_train,
    get_initial_conditions,
    setup_mlp,
    setup_time_masks_u0_curriculum
)
from src.model_training.training_util import get_save_epochs
from src.ude_problem import UDEProblem


# ================================================================
#            GET CONFIG, SET PRECISION, GET PRNG KEYS
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
#                SET UP IVs, DATA, WINDOWS, MASKS
# ================================================================
# Get initial conditions
u0 = get_initial_conditions(conf)

# Setup a mock MLP
mlp = setup_mlp(conf, u0, key_mlp, key_init)

# Get data and functions for training
(
    train_args, val_args, loss_dict, option_args, plot_args, rescaling_args
) = get_data_and_funcs_for_train(
    conf, true_dynamic_parameters[conf.dynamical_model].shape, u0
)

# Set up SaveAt windows, binary masks and initial values
(
    t_save_windows,
    masks,
    initial_values,
    curriculum_stage_dict
 ) = setup_time_masks_u0_curriculum(
    conf=conf,
    train_args=train_args,
    u0_guess=train_args.u0,
    effective_state_dimension=mlp.layers[0].in_features,
)

# ================================================================
#                    LOAD MODEL CHECKPOINTS
# ================================================================

# Compute dict with curriculum stages as keys and corresponding save epochs as values
if not conf.curriculum_learning:
    num_save_models = 10
    save_epochs = {1: get_save_epochs(conf.num_epochs_adam, num_save_models)}
else:
    # Curriculum save points: start and end of each stage
    save_epochs = {
        stage: [
            curriculum_stage_dict[stage]["start"],
            curriculum_stage_dict[stage]["start"] +
            (curriculum_stage_dict[stage]["end"] - curriculum_stage_dict[stage]["start"]) //2,
            curriculum_stage_dict[stage]["end"]-1
        ]
        for stage in curriculum_stage_dict
    }
    model_save_steps_start = [
        curriculum_stage_dict[stage]["start"] for stage in curriculum_stage_dict
    ]


model_save_subdir = (
    model_save_dir /
    conf.dynamical_model /
    str(conf.experiment_name) /
    conf.training_setting
)

# Load models into dict with curriculum stages as outer keys and epochs as inner keys
model_dict = {
    cl_stage: {
        epoch: UDEProblem.load(
            model_save_subdir /
            "checkpoints" /
            (conf.__str__() + f"_ep{epoch}_CL{cl_stage}.eqx")
        )
        for epoch in save_epochs[cl_stage]
    }
    for cl_stage in save_epochs.keys()
}

for mode in ["local", "global"]:
    convexity_results = test_curriculum_stage_convexity(
        model_dict,
        conf,
        curriculum_stage_dict,
        train_args.data,
        t_save_windows,
        masks,
        loss_dict["loss_fn"],
        which="full",
        num_dirs=100,
        mode=mode
    )

    plot_convexity_fractions(convexity_results, mode=mode)

visualize_hessian_eigvec_landscape(
    model_dict,
    conf,
    t_save_windows,
    masks,
    train_args.data,
    curriculum_stage_dict,
    base_loss_fn=loss_dict["base_loss_fn"],
    # loss_fn is already a partial with rescaling_args nad max_steps
    loss_fn=loss_dict["loss_fn"],
    # hessian_fn needs to be turned into a partial
    hessian_fn=partial(
        compute_hess_array,
        rescaling_args=rescaling_args,
        max_steps=max_iters_diffrax[conf.dynamical_model]
    ),
)

# Lotka-Volterra-specific analysis given ground truth dynamic parameters (alpha, delta)
if conf.dynamical_model == "lotka_volterra" and conf.training_setting == "UDE":
    plot_lotka_volterra_dynamic_param_trajectory(
        model_dict,
        true_dynamic_parameters[conf.dynamical_model],
        rescaling_args,
    )

    visualize_lv_loss_around_true_dyn_params(
        model_dict,
        conf,
        curriculum_stage_dict,
        train_args.data,
        t_save_windows,
        masks,
        loss_dict["loss_fn"],
        true_dynamic_parameters[conf.dynamical_model],
        rescaling_args,
        plot_in_log=True
    )

    visualize_lv_loss_on_dyn_params_per_stage(
        model_dict,
        conf,
        curriculum_stage_dict,
        train_args.data,
        t_save_windows,
        masks,
        loss_dict["loss_fn"],
        true_dynamic_parameters[conf.dynamical_model],
        rescaling_args,
        plot_in_log=True
    )

print("Analysis complete.")
