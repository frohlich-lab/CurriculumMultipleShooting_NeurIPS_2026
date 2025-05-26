import fire
import jax.random as jr
import jax.numpy as jnp
import os
import pickle

from dataclasses import fields, replace
from jax import config
from pathlib import Path
from src.common import (Conf, EarlyStoppingParams, model_save_dir, results_dir,
                        state_dims, true_dynamic_parameters)
from src.model_training.training_setup import get_data_and_funcs_for_train, get_initial_conditions, setup_mlp
from src.model_training.training_util import sample_dynamical_params, get_sindy_formula, generate_epochs_dict
from src.model_training.landscape_plotting_util import analyse_lv_loss_landscape_dynparams, get_ude_param_landscape
from src.ude_problem import UDEProblem
from training import train
from training_configuration import N_EPOCHS, MAX_ITER_OPTX_FINETUNE
from tqdm.contrib.itertools import product  # directly using itertools.product progress bar wrapper

conf = fire.Fire(Conf)

# Technically Float32 should work, but for consistency between the benchmarks
# I think we should use Float64
config.update("jax_enable_x64", True)

# Define results path
results_path = Path(
    results_dir /
    conf.dynamical_model /
    "two_stage_training"
)
results_path.mkdir(parents=True, exist_ok=True)

# Define model save path
model_save_path = (
    model_save_dir /
    conf.dynamical_model /
    "two_stage_training"
)
model_save_path.mkdir(parents=True, exist_ok=True)


for job, second_curriculum_learning in product(range(10), [True, False], position=0, leave=True): # brutally horrible way to do this, but need to get some of these done
    conf = replace(conf, multistart_job=job, second_curriculum_learning=second_curriculum_learning)
    print("Now running job", job, "with second curriculum learning set to", second_curriculum_learning)
    # Set seed for sampling dynamical parameters
    seed = conf.multistart_job

    # Key
    key = jr.PRNGKey(seed)
    key_mlp, key_init = jr.split(key, 2)

    # Get initial conditions
    u0 = get_initial_conditions(conf)

    # ================================================================
    #                         SET UP MLP
    # ================================================================
    mlp = setup_mlp(conf, u0, key_mlp, key_init)

    # ================================================================
    #                   SET UP UDE PROBLEM TO TRAIN
    # ================================================================
    # Use true dynamic params for first stage
    dynamic_parameters = {
        dynamical_model: true_dynamic_parameters[dynamical_model]
        if conf.dynamic_params_scale == 'linear'
        else jnp.log10(true_dynamic_parameters[dynamical_model])
        for dynamical_model in true_dynamic_parameters.keys()
    }

    # Get data and functions for training
    train_args, val_args, loss_args, option_args = get_data_and_funcs_for_train(
        conf, true_dynamic_parameters[conf.dynamical_model].shape
    )
    train_data, t_save_train, t_span_train = train_args
    validation_data, t_save_validation, t_span_validation = val_args
    loss_fn, loss_grad_fn, loss_optx = loss_args
    optx_config, options = option_args

    ude_problem = UDEProblem(
        dynamical_model=conf.dynamical_model,
        dynamic_parameters=dynamic_parameters[conf.dynamical_model],
        mlp=mlp,
        options=options,
        state_dimension=state_dims[conf.dynamical_model],
        training_horizon=train_data.shape[0],
        shooting_mode=conf.shooting_mode,
        shooting_window_length=conf.shooting_window_length,
        shooting_iv_init=conf.shooting_iv_init,
        u0_guess=u0,
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
        serialise_curriculum_models=False,
    )
    # Define model save path
    subconf_fields = [
        "mlp_layer_width", "mlp_depth", "init_scale", "use_bias", "use_final_bias",
        "activation_fn_name", "final_activation_fn_name",
        "optimiser", "max_lrate", "lrate_span", "lrate_decay", "warmup_fct", "opt_steps", "opt_mult",
        "multistart_deviation", "multistart_job",
    ]
    subconf_string = "__".join([str(getattr(conf, field.name)) for field in fields(conf) if field.name in subconf_fields])
    model_file = model_save_path / (subconf_string + f"__step1MLP.eqx")
    if os.path.exists(model_file):
        print(f"Model file already exists, loading and skipping training.")
        trained_model = UDEProblem.load(model_file)
    else:
        print("Training MLP...")
        trained_model = train(
            model=ude_problem,
            u0=u0,
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
            n_epochs=N_EPOCHS,  # only train using ADAM
            max_iter_optx_finetune=0,
            model_save_dir=model_save_dir,
        )
        # Serialise model
        trained_model.save(model_file)

    # Get SINDy formulas for whole UDE and trained MLP
    get_sindy_formula(
        model=trained_model,
        u0=u0,
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
    # Decide whether to train dynamical params with curriculum learning or not
    conf = replace(
        conf,
        curriculum_learning=conf.second_curriculum_learning,
        freeze_mechanistic_params_adam=False,
        freeze_mechanistic_params_optx=False,
        freeze_nn_adam=True,
        freeze_nn_optx=True,
        serialise_curriculum_models=True,
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

    initial_dynamic_params = reinit_model.func.dynamic_parameters  # on log10 scale
    print(f"Initial dynamic parameters: {jnp.power(10, initial_dynamic_params) if conf.dynamic_params_scale == 'log10' else initial_dynamic_params}")

    # Define model save path
    subconf_string = "__".join(
        [str(getattr(conf, field.name)) for field in fields(conf) if field.name in subconf_fields])
    landscape_model_file = model_save_path / (
                subconf_string + f"__step2dynpar__CL2{conf.second_curriculum_learning}.eqx"
    )
    opt_dyn_params_file = model_save_path / (
        subconf_string + f"__step2dynpar__CL2{conf.second_curriculum_learning}_pars.pkl"
    )
    if not os.path.exists(landscape_model_file) or not os.path.exists(opt_dyn_params_file):
        # Re-train
        print("Training dynamic parameters...")
        training_output = train(
            model=reinit_model,  # initialise with trained model with new dynamic params
            u0=u0,
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
            n_epochs=N_EPOCHS,
            max_iter_optx_finetune=MAX_ITER_OPTX_FINETUNE,
            model_save_dir=model_save_dir,
        )

        if conf.second_curriculum_learning:
            # Unpack output
            final_model, curriculum_models = training_output
            # Get dynamic parameters at the beginning and end of each curriculum step
            cl_epochs = generate_epochs_dict(sorted(list(curriculum_models.keys())))
            opt_dynamic_params = {
                cl_step: {
                    timepoint: curriculum_models[cl_epochs[cl_step][timepoint]].func.dynamic_parameters
                    for timepoint in ["start", "end"]
                }
                for cl_step in range(conf.num_curriculum_steps)
            }

            # Get optimised parameters from last ADAM model and, if needed, log10-transform them
            landscape_model = curriculum_models[sorted(list(curriculum_models.keys()))[-1]]
        else:
            # Unpack output
            landscape_model = training_output
            # Get dynamic parameters at the beginning and end of the whole training
            opt_dynamic_params = {0: {"start": initial_dynamic_params, "end": landscape_model.func.dynamic_parameters}}

        # Serialise landscape model and dynamic parameters history
        print("Serialising final model and dynamic parameters history...")
        landscape_model.save(landscape_model_file)
        with open(opt_dyn_params_file, "wb") as f:
            pickle.dump(opt_dynamic_params, f)

    else:
        print(f"Both landscape model and dynamic parameters history file exist, loading and skipping retraining..")
        with open(opt_dyn_params_file, "rb") as f:
            opt_dynamic_params = pickle.load(f)
        landscape_model = UDEProblem.load(landscape_model_file)

    # Print out final dynamic parameters and ground truth
    final_dynamic_params = landscape_model.func.dynamic_parameters
    print(f"Final dynamic parameters: {jnp.power(10, final_dynamic_params) if conf.dynamic_params_scale == 'log10' else final_dynamic_params}")
    print(f"Ground truth parameters: {true_dynamic_parameters[conf.dynamical_model]}")  # these are always on linear scale

    landscape_results_file = results_path / (subconf_string + f"_CL{conf.second_curriculum_learning}_job{conf.multistart_job}_step2dynpar.csv")
    if not os.path.exists(landscape_results_file):
        print("Performing loss landscape analysis vs dynamic parameters...")
        # Perform loss landscape analysis
        landscape_dynparams_df = analyse_lv_loss_landscape_dynparams(
            udeprob=landscape_model,
            u0=u0,  # augment initial condition for network regularisation
            train_data=train_data,
            t_save_train=t_save_train,
            t_span_train=t_span_train,
            validation_data=validation_data,
            t_save_validation=t_save_validation,
            t_span_validation=t_span_validation,
            conf=conf,
            N_EPOCHS=N_EPOCHS,
            opt_dynamic_params=opt_dynamic_params,
            do_plots=True,
            fig_name=f"two_stage_loss_train_CL{conf.second_curriculum_learning}_job{conf.multistart_job}",
            fig_fmts=["pdf", "svg"],
            num_points=200,
        )
        # Save landscape dataframe to file (CSV
        landscape_dynparams_df.assign(multistart_job=conf.multistart_job)
        landscape_dynparams_df.to_csv(
            landscape_results_file,
            index=False
        )
    else:
        print("Loss landscape analysis file already exists, skipping...")


    if conf.second_curriculum_learning:
        nn_params_file = results_path / (subconf_string + f"__step2__CL2{conf.second_curriculum_learning}.csv")
        if not os.path.exists(nn_params_file):
            print("Performing loss landscape analysis vs MLP parameters...")
            nn_param_landscape_df = get_ude_param_landscape(
                udeprob=landscape_model,
                u0=u0,
                train_data=train_data,
                t_save_train=t_save_train,
                t_span_train=t_span_train,
                validation_data=validation_data,
                t_save_validation=t_save_validation,
                t_span_validation=t_span_validation,
                conf=conf,
                key_list=[jr.PRNGKey(conf.multistart_job)],  # single pair of random directions
                num_dirs=2,
                start_stop=[-1, 1],
                num_points=100,
                which_params=["MLP"],
                do_plots=True,
                fig_name=f"two_stage_step2nnparams_job{conf.multistart_job}",
            )
            nn_param_landscape_df.to_csv(
                nn_params_file,
                index=False
            )
        else:
            print("Loss landscape analysis file already exists, skipping...")
