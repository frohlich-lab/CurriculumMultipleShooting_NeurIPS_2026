import diffrax
import equinox as eqx
# import equinox.debug
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
# import matplotlib.pyplot as plt
import optax
import pandas as pd
# import subprocess
import wandb
import warnings

from src.common import (Conf, dynamic_param_labels, EarlyStoppingParams, hessian_metrics_wandb,
                        plots_dir, results_dir, TrainValArgs)
from dataclasses import replace
from diffrax import SaveAt
from jaxtyping import Array, Float, PyTree
from pathlib import Path
from src.model_training.training_util import (check_and_progress_curriculum, configure_early_stopping,
                                              generate_incremental_arrays, generate_log_epochs,
                                              get_opt_opt_state_params, get_scheduler,
                                              initialise_wandb, log_model_stats, log_plots,
                                              make_grads_finite, max_grad_norm, get_hessian_metrics,
                                              MetricHandler, transform_dynamic_params, augment_u0)
from src.model_training.optimistix_training import train_optimistix
from src.model_training.fides_training import train_fides
from src.model_training.hessian import get_hessian, apply_hessian_preconditioning_to_grads
from src.ude_problem import UDEProblem
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Some FutureWarning with unhashable DynamicJaxprTracer object in the diffeq solve/call
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_fit_dfs(
        prob: UDEProblem,
        u0,
        t_save,
        t_span,
        data: jnp.ndarray,
        dataset: str,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    # Model prediction
    if dataset == "train":
        if isinstance(t_save, SaveAt):
            t_save = t_save.subs.ts
        sol = prob.solve_single_shooting(diffrax.Tsit5(), u0, t_save)
    else:
        sol = prob.solve_single_shooting_validation(diffrax.Tsit5(), u0, t_save, t_span)

    if prob.dynamical_model == "lotka_volterra":
        df_pred = pd.DataFrame({"time": sol.ts,
                                "prey": sol.ys[:, 0],
                                "predator": sol.ys[:, 1],
                                })
        df_obs = pd.DataFrame({"time": sol.ts,
                               "prey": data[:, 0],
                               "predator": data[:, 1],
                               })
    elif prob.dynamical_model == "sir":
        df_pred = pd.DataFrame({"time": sol.ts,
                                "Infected": jnp.log(jnp.absolute(sol.ys[:, 1] + sol.ys[:, 3])),
                                "Recovered + dead": jnp.log(jnp.absolute(sol.ys[:, 2])),
                                })
        # Observed data
        infected, dead, recovered = data[:, 0], data[:, 1], data[:, 2]
        df_obs = pd.DataFrame({"time": sol.ts,
                               "Infected": jnp.log(infected),
                               "Recovered + dead": jnp.log(jnp.absolute(recovered + dead)),
                               })
    else:
        raise ValueError(f"Undefined dynamical model {prob.dynamical_model}.")
    if (prob.shooting_mode == "single") or (dataset != "train"):
        return df_pred, None, df_obs
    else:
        if prob.shooting_mode == "multiple":
            # TODO @GiacomoFabrini -- need to fix u0/t_save - provided vs internal?
            sol = prob.solve_multiple_shooting(
                diffrax.Tsit5(),
                prob.initial_values,
                prob.t_save_windows,
            )
        else:
            raise ValueError(f"Undefined shooting mode {prob.shooting_mode}.")
        if prob.dynamical_model == "lotka_volterra":
            df_pred_train = pd.concat([
                pd.DataFrame({"time": sol.ts[i],
                              "prey": sol.ys[i, :, 0],
                              "predator": sol.ys[i, :, 1],
                              }).assign(batch=i + 1)
                for i in range(sol.ys.shape[0])  # batch/shooting-window dimension
            ])
        elif prob.dynamical_model == "sir":
            df_pred_train = pd.concat([
                pd.DataFrame({"time": sol.ts[i],
                              "Infected": jnp.log(jnp.absolute(sol.ys[i, :, 1] + sol.ys[i, :, 3])),
                              "Recovered + dead": jnp.log(jnp.absolute(sol.ys[i, :, 2])),
                              }).assign(batch=i + 1)
                for i in range(sol.ys.shape[0])  # batch/shooting-window dimension
            ])
        else:
            raise ValueError(f"Undefined dynamical model {prob.dynamical_model}.")
        return df_pred, df_pred_train, df_obs


# PLOTTING HELPER FUNCTION
def log_fit_plots(
        model: UDEProblem,
        u0: jnp.ndarray,
        t_save_train: SaveAt, t_span_train: List, training_data: jnp.ndarray,
        t_save_val: SaveAt, t_span_val: List, validation_data: jnp.ndarray,
        conf: Conf,
        colors: List[str],
        epoch: int,
        plot_name: str,
        plot_subdir: str,
):
    # Log plots again after further training with optimistix/BFGS
    for dataset, t_save, t_span, data in zip(
            ["train", "validation"],
            [t_save_train, t_save_val],
            [t_span_train, t_span_val],
            [training_data, validation_data]
    ):
        # TODO @GiacomoFabrini check this and whether more checks are needed!
        if (len(u0) > model.func.state_dimension) and (conf.node_augmentation == 0):
            assert model.func.regularise_network == True, ("Regularisation of the network is required, u0 has more "
                                                           "elements than state dimension.")
        df_predict, df_predict_train, df_obs = get_fit_dfs(
            model, u0, t_save, t_span, data, dataset,
        )
        plot_dir = plots_dir / conf.dynamical_model / plot_subdir
        plot_dir.mkdir(exist_ok=True, parents=True)
        log_plots(
            [df_predict, df_predict_train],
            df_obs,
            dataset,
            model.shooting_mode,
            colors,
            epoch,
            f"{dataset}_{plot_name}",
            save_plot=conf.save_plots,
            plot_dir=plot_dir,
            plot_fmt="svg",
        )


# NN OPTIMISATION STEP
@eqx.filter_jit
def make_step(
        params: Union[PyTree, UDEProblem],
        static: Union[PyTree, UDEProblem],
        opt: optax.GradientTransformation,
        opt_state: PyTree,
        input_data: Float[Array, '...'],  # TODO @GiacomoFabrini fix input data shape
        u0: jnp.ndarray,
        t_save: SaveAt,
        t_span: List,
        conf: Conf,
        loss_grad_fn: Callable,
        compute_hessian_metrics: bool,
):
    # TODO @GiacomoFabrini - check whether LookAhead still works with newer changes! If not, discuss and decide whether to keep or remove
    if conf.use_lookahead:
        opt_params = params.fast
    else:
        opt_params = params
    (loss_value, reg_loss_dict), grads = loss_grad_fn(
        prob_params=opt_params,
        prob_static=static,
        u0=u0,
        t_save=t_save,
        t_span=t_span,
        data=input_data,
        mlp_reg_exp=conf.mlp_reg_exp,
        mlp_reg_strength=conf.mlp_reg_strength,
        output_reg=conf.output_reg,
        multiple_shooting_penalty=conf.multiple_shooting_penalty,
        shooting_mode=conf.shooting_mode,
    )
    grads = make_grads_finite(grads)
    # Initialise hessian metrics with NaNs
    hessian_metrics = {
        metric: jnp.nan
        for metric in hessian_metrics_wandb.keys()
    }
    if conf.hessian_precondition:
        args = (u0, t_save, t_span, input_data)
        grads, updated_hessian_metrics = apply_hessian_preconditioning_to_grads(
            grads, opt_params, static, args, conf
        )
    elif compute_hessian_metrics:
        args = (u0, t_save, t_span, input_data)
        hessian_metrics = get_hessian_metrics(
            get_hessian(opt_params, static, args, conf),
            return_eigs_svs=False,
        )
    updates, opt_state = opt.update(grads, opt_state, params)
    params = eqx.apply_updates(params, updates)
    return params, opt_state, loss_value, reg_loss_dict, grads, hessian_metrics


# MAIN TRAINING FUNCTION
def train(
        model: UDEProblem,
        train_args: TrainValArgs,
        val_args: TrainValArgs,
        curriculum_params: Optional[dict[int, Any]],
        conf: Conf,
        early_stopping_params: EarlyStoppingParams,
        loss_dict: Dict[str, Callable],
        optx_config: Dict,
        max_iter_optx_mechanistic: int,
        n_epochs: int,
        max_iter_optx_finetune: int,
        model_save_dir: Path,
        plot_args: dict,
) -> Union[UDEProblem, Tuple[UDEProblem, Dict[int, UDEProblem]]]:
    """
    Trains the UDEmodel (`model`).

    Parameters
    ----------
    :param model:
        UDEProblem to train.
    :param train_args:
        Training arguments (u0, t_save, t_span, data) in TrainValArgs class.
    :param val_args:
        Validation arguments (u0, t_save, t_span, data) in TrainValArgs class.
    :param curriculum_params: Optional[dict]
        Dictionary of curriculum learning parameters.
    :param conf:
        configuration (hyperparameters).
    :param early_stopping_params:
        Dictionary of parameters for Early-Stopping.
    :param loss_dict:
        Dictionary of loss functions.
    :param optx_config:
        Configuration for optimistix solver (rtol, atol).
    :param max_iter_optx_mechanistic:
        Maximum number of iterations for initial fitting of mechanistic parameters (dynamical parameters).
    :param n_epochs:
        Number of training iterations (epochs).
    :param max_iter_optx_finetune:
        Maximum number of iterations for further training via optimistix.
    :param model_save_dir:
        Filepath of directory to save the trained models.
    :param plot_args:
        Arguments for plotting train/val data.
    """

    # Check training setting in configuration
    if conf.training_setting not in ["UDE", "NODE"]:
        # if not properly set, set to default value of UDE
        conf = replace(conf, training_setting="UDE")

    # Check curriculum learning (CL) settings
    if not conf.curriculum_learning:
        conf = replace(conf, serialise_curriculum_models=False)  # cannot serialise CL models if not using CL

    # Initialise wandb and define metrics
    initialise_wandb(
        conf=conf,
        early_stopping_params=early_stopping_params,
        model=model,
    )

    # Initialise variables
    epoch, nn_epoch = 0, 0
    best_model, loss_val_min = model, jnp.inf

    # Unpack training and validation args
    u0, training_data, t_save_train, t_span_train = train_args.u0, train_args.data, train_args.t_save, train_args.t_span
    u0_val, validation_data, t_save_val, t_span_val = val_args.u0, val_args.data, val_args.t_save, val_args.t_span

    # Create filter_spec allowing for training of mechanistic params only or neural network only
    filter_spec_freeze_mechanistic = jtu.tree_map(lambda x: eqx.is_array(x), model)
    # filter_spec which freezes dynamic (mechanistic) parameters
    filter_spec_freeze_mechanistic = eqx.tree_at(
        lambda x: x.func.dynamic_parameters,
        filter_spec_freeze_mechanistic,
        replace=jax.tree_map(lambda _: False, filter_spec_freeze_mechanistic.func.dynamic_parameters)
    )
    # filter_spec which freezes the neural network
    filter_spec_freeze_nn = jtu.tree_map(lambda x: False, model)
    filter_spec_freeze_nn = eqx.tree_at(
        lambda x: x.func.dynamic_parameters,
        filter_spec_freeze_nn,
        replace=jax.tree_map(lambda _: True, filter_spec_freeze_nn.func.dynamic_parameters)
    )

    # Define colours for plots
    if conf.dynamical_model == "lotka_volterra":
        colors = ['red', 'black']
    elif conf.dynamical_model == 'sir':
        colors = ['red', 'black']
    else:
        raise ValueError(f"Undefined dynamical model {conf.dynamical_model}.")

    # Save the initialised MLP from model
    initialised_mlp = model.func.mlp

    # TODO @GiacomoFabrini - DISABLED THIS - DO WE NEED IT?!
    if conf.curriculum_loss_landscape_BFGS and conf.curriculum_learning:
        # Generate incremental training data for NN training - curriculum learning or not (if not, full data at each epoch)
        incremental_training_data, incremental_train_save, incremental_t_span, steps = generate_incremental_arrays(
            data=training_data,
            t_save=t_save_train,
            t_span=t_span_train,
            n_epoch=n_epochs,
            max_curriculum_epochs=conf.max_curriculum_epochs,
            curriculum_params=curriculum_params,
        )
        # For each number of training datapoints, i.e. at each step in the curriculum learning ladder,
        # train the model with optimistix (BFGS)
        for curriculum_step in range(conf.num_curriculum_steps):
            print(len(incremental_training_data[steps[curriculum_step]]))
            # Reset starting epoch
            epoch = 0
            # NN learnable, mechanistic parameters can be learnable or frozen depending on conf.freeze_mechanistic_params_landscape_BFGS
            if conf.second_opt == "optx":
                cl_model, sol, static, loss_train, loss_val, epoch, _ = train_optimistix(
                    model=model,
                    filter_spec=filter_spec_freeze_mechanistic if conf.freeze_mechanistic_params_landscape_BFGS
                        else eqx.is_array,
                    conf=conf,
                    optx_config=optx_config,
                    base_loss=loss_dict["base_loss_fn"],
                    loss_optx=loss_dict["loss_optx"],
                    val_base_loss=loss_dict["val_base_loss_fn"],
                    val_loss_optx=loss_dict["val_loss_optx"],
                    epoch=epoch,
                    num_epochs=10000,  # TODO @GiacomoFabrini - do we want to include an extra hyperparam for this?
                    u0_train=u0,
                    u0_val=u0_val,
                    t_save_train=incremental_train_save[steps[curriculum_step]],
                    t_span_train=incremental_t_span[steps[curriculum_step]],
                    training_data=incremental_training_data[steps[curriculum_step]],
                    t_save_val=t_save_val,
                    t_span_val=t_span_val,
                    validation_data=validation_data,
                    frozen_mechanistic=conf.freeze_mechanistic_params_landscape_BFGS,
                    frozen_nn=False,
                    zero_out_mlp=False,
                    wandb_log=False,
                )
            else:
                cl_model, sol, static, loss_train, loss_val, epoch, _ = train_fides(
                    model=model,
                    filter_spec=filter_spec_freeze_mechanistic if conf.freeze_mechanistic_params_landscape_BFGS
                    else eqx.is_array,
                    conf=conf,
                    loss_grad_fn=loss_dict["loss_grad_fn"],
                    epoch=epoch,
                    max_iter=10000,  # TODO @GiacomoFabrini - do we want to include an extra hyperparam for this?
                    u0_train=u0,
                    u0_val=u0_val,
                    t_save_train=incremental_train_save[steps[curriculum_step]],
                    t_span_train=incremental_t_span[steps[curriculum_step]],
                    training_data=incremental_training_data[steps[curriculum_step]],
                    t_save_val=t_save_val,
                    t_span_val=t_span_val,
                    validation_data=validation_data,
                    loss_fn=loss_dict["loss_fn"],
                    val_loss_fn=loss_dict["val_loss_fn"],
                    zero_out_mlp=False,
                )
            # Compute training loss on the full training set
            whole_train_loss,  _ = loss_dict["loss_fn"](
                prob_params=sol.value,
                prob_static=static,
                u0=u0,
                t_save=t_save_train,
                t_span=t_span_train,
                data=training_data,
                mlp_reg_exp=conf.mlp_reg_exp,
                mlp_reg_strength=conf.mlp_reg_strength,
                output_reg=conf.output_reg,
            )
            # Log metrics to wandb
            wandb.log(
                {
                    "loss_train": loss_train,
                    "full_train_loss": whole_train_loss,
                    "loss_val": loss_val,
                    "num_train_samples": len(incremental_training_data[steps[curriculum_step]]),
                },
                step=curriculum_step,
            )
            # Log fit plots to wandb
            log_fit_plots(
                cl_model, u0,
                t_save_train, t_span_train, training_data,  # plot for full training set just for reference
                t_save_val, t_span_val, validation_data,
                conf, colors, curriculum_step,
                f"BFGS_local_minima_{curriculum_step+1}",
                "BFGS_local_minima",
            )
        return model

    results_dfs = []
    # STAGE 1: fit mechanistic model parameters -- zero-out and freeze neural network
    if max_iter_optx_mechanistic > 0:
        wandb_log = True
        if conf.second_opt == "optx":
            model, sol, static, _, loss_val, epoch, partial_results_df = train_optimistix(
                model=model,
                filter_spec=filter_spec_freeze_nn,
                conf=conf,
                optx_config=optx_config,
                base_loss=loss_dict["base_loss_fn"],
                loss_optx=loss_dict["loss_optx"],
                val_base_loss=loss_dict["val_base_loss_fn"],
                val_loss_optx=loss_dict["val_loss_optx"],
                epoch=epoch,
                num_epochs=max_iter_optx_mechanistic,
                u0_train=u0,
                u0_val=u0_val,
                t_save_train=t_save_train,
                t_span_train=t_span_train,
                training_data=training_data,
                t_save_val=t_save_val,
                t_span_val=t_span_val,
                validation_data=validation_data,
                frozen_mechanistic=False,
                frozen_nn=True,
                zero_out_mlp=True,
                wandb_log=wandb_log,
                wandb_log_final=False,
            )
        else:
            model, sol, static, _, loss_val, epoch, partial_results_df = train_fides(
                model=model,
                filter_spec=filter_spec_freeze_nn,
                conf=conf,
                loss_grad_fn=loss_dict["loss_grad_fn"],
                epoch=epoch,
                max_iter=max_iter_optx_mechanistic,
                u0_train=u0,
                u0_val=u0_val,
                t_save_train=t_save_train,
                t_span_train=t_span_train,
                training_data=training_data,
                t_save_val=t_save_val,
                t_span_val=t_span_val,
                validation_data=validation_data,
                loss_fn=loss_dict["loss_fn"],
                val_loss_fn=loss_dict["val_loss_fn"],
                zero_out_mlp=True,
                wandb_log_final=wandb_log,
            )
        if wandb_log:  # if not, partial_results will be None
            # Add conf/hyperparameters details
            results_dfs.append(partial_results_df.assign(**conf.__dict__))

        # Update best_model
        if loss_val < loss_val_min:
            best_model = model

        # Plot eigenvalue and singular value spectrum
        # hessian = get_hessian(sol.value, static, (u0, t_save_train, t_span_train, training_data), conf)
        # hessian_metrics, eigs, svs = get_hessian_metrics(hessian, return_eigs_svs=True)
        # eig_sv_plot(
        #     eigs, svs,
        #     conf,
        #     epoch,
        # )

        # Log plots again after further training with optimistix/BFGS
        log_fit_plots(
            model=model, **plot_args,
            conf=conf,
            colors=colors, epoch=epoch,
            plot_name=f"mechanistic_{conf.second_opt}",
            plot_subdir=f"mechanistic_{conf.second_opt}",
        )

        # Increment epoch before moving to next stage
        epoch +=1


    # STAGE 2: Train with ADAM/ADAM-W -- freeze mechanistic model parameters
    if n_epochs > 0:
        # Restore the previously initialised MLP + enable network contribution regularisation
        model = model.replace_mlp(
            new_mlp=initialised_mlp,
            regularise_network=True,
        )
        best_model = best_model.replace_mlp(
            new_mlp=initialised_mlp,
            regularise_network=True,
        )
        # Add initial zero as starting condition for the 2-norm of the neural network contribution
        u0 = augment_u0(u0)
        u0_val = augment_u0(u0_val)
        plot_args["u0"] = augment_u0(plot_args["u0"])

        # Initialise CL stage variable
        cl_stage = 1  # initialise CL stage

        # Depending on conf settings, freeze mechanistic params/neural network or keep everything learnable
        if conf.freeze_mechanistic_params_adam:
            adam_filter_spec = filter_spec_freeze_mechanistic
        elif conf.freeze_nn_adam:
            adam_filter_spec = filter_spec_freeze_nn
        else:
            adam_filter_spec = eqx.is_array
        params, static = eqx.partition(model, adam_filter_spec)

        # Check Early-stopping parameters have been set correctly and instantiate early stopper
        early_stopper = configure_early_stopping(early_stopping_params)

        # Generate regularly log-spaced epochs for early-stopping evaluation + model stat logging - from DMMs
        log_epochs = generate_log_epochs(n_epochs=n_epochs, num_samples=100, min_dist=5)

        # Define epochs at which to save curriculum-step models: just before increasing number of datapoints or last epoch
        if conf.curriculum_learning:
            model_save_steps_start = [
                curriculum_params[stage]["start"]
                for stage in curriculum_params.keys()
            ]  # start of CL stages
            model_save_steps_end = [
                curriculum_params[stage]["end"]
                for stage in curriculum_params.keys()
            ]  # end of CL stages
            models_curriculum_stage_start = {}
            models_curriculum_stage_end = {}

        # Get schedule
        schedule = get_scheduler(conf=conf, n_epochs=n_epochs)
        # Initialise optimiser, optimiser state + update params if using lookahead
        opt, opt_state, params = get_opt_opt_state_params(conf, params, schedule)


        # Training loop
        metric_handler = MetricHandler(patience=5)
        ivs = []
        for nn_epoch in range(n_epochs):
            # Compute validation loss prior to epoch update
            if conf.use_dropout:
                # Get a copy of the model for inference
                inference_model = model.replace_mlp(
                    new_mlp=model.func.mlp,
                    switch_to_inference=True,
                )
            else:
                # no need for difference in training/inference behaviour
                inference_model = model
            inference_params, inference_static = eqx.partition(inference_model, adam_filter_spec)
            # compute validation loss
            loss_val, _ = loss_dict["val_loss_fn"](
                prob_params=inference_params,
                prob_static=inference_static,
                u0=u0_val,
                t_save=t_save_val,
                t_span=t_span_val,
                data=validation_data,
            )

            # Keep track of best performing model on validation dataset (and corresponding loss)
            if loss_val < loss_val_min:
                best_model = inference_model
                loss_val_min = loss_val


            if conf.shooting_mode == "multiple":
                ivs.append(params.initial_values)

            if conf.curriculum_learning:
                params, static = check_and_progress_curriculum(
                    params, static, adam_filter_spec,
                    nn_epoch, model_save_steps_start, conf.shooting_mode,
                    curriculum_params
                )

            params, opt_state, loss_train, reg_loss_dict, grads, hessian_metrics = make_step(
                params=params,
                static=static,
                opt=opt,
                opt_state=opt_state,
                input_data=training_data,
                u0=u0,
                t_save=t_save_train,
                t_span=t_span_train,
                conf=conf,
                loss_grad_fn=loss_dict["loss_grad_fn"],
                compute_hessian_metrics=(conf.compute_hessian_metrics and (nn_epoch in log_epochs)),
            )

            if metric_handler.check_whether_should_break(loss_train):
                break

            # Recombine dynamic and static into full model
            model = eqx.combine(
                params.fast if conf.use_lookahead else params,
                static,
            )

            # Save model at the start/end of each curriculum learning stage
            if conf.serialise_curriculum_models:
                if nn_epoch in model_save_steps_start:
                    models_curriculum_stage_start[nn_epoch] = inference_model
                elif nn_epoch in model_save_steps_end:
                    models_curriculum_stage_end[nn_epoch] = inference_model

            partial_results_df = pd.DataFrame(
                [{
                    "optimiser": conf.optimiser,
                    "epoch": epoch+nn_epoch,
                    # Losses
                    "loss_train": loss_train,
                    "loss_val": loss_val,
                    # Training metrics
                    "learning_rate": schedule(nn_epoch),
                    "num_train_samples": curriculum_params[cl_stage]["length"] if conf.curriculum_learning \
                        else len(training_data),
                    "max_grad_norm": max_grad_norm(grads),
                    # Dynamic parameters
                    **{
                        param: transform_dynamic_params(
                            inference_model.func.dynamic_parameters, conf
                        )[i]
                        for i, param in enumerate(dynamic_param_labels[conf.dynamical_model])
                    },
                }]
            )
            # log loss_train, loss_val, learning rate, param values and grads on log-spaced epochs
            if (nn_epoch in log_epochs) or (nn_epoch == 0):
                wandb.log(
                    {
                        "loss_train": loss_train,
                        "loss_val": loss_val,
                        "learning_rate": schedule(nn_epoch),
                        "num_train_samples": curriculum_params[cl_stage]["length"] if conf.curriculum_learning \
                            else len(training_data),
                        "max_grad_norm": max_grad_norm(grads),
                    },
                    step=epoch+nn_epoch
                )
                if jnp.isfinite(loss_train):
                    wandb.log(
                        log_model_stats(
                            model, grads, conf,
                            frozen_mechanistic=conf.freeze_mechanistic_params_adam,
                            frozen_nn=conf.freeze_nn_adam
                        ),
                        step=epoch+nn_epoch
                    )

                if conf.compute_hessian_metrics:
                    # Log Hessian metrics
                    wandb.log(
                        {
                            **hessian_metrics
                        },
                        step=epoch+nn_epoch
                    )
                    # Add Hessian metrics to partial results
                    partial_results_df = partial_results_df.assign(**hessian_metrics)

                # Log discontinuity penalty if applied
                if (conf.multiple_shooting_penalty > 0) and (conf.shooting_mode == "multiple"):
                    wandb.log(
                        {
                            "discontinuity_penalty": reg_loss_dict["discontinuity_penalty"]
                        },
                        step=epoch+nn_epoch
                    )
                    partial_results_df = partial_results_df.assign(
                        discontinuity_penalty=reg_loss_dict["discontinuity_penalty"],
                    )
                # Log regularisation term if applied
                if conf.mlp_reg_strength > 0:
                    wandb.log(
                        {
                            "regularisation_loss": reg_loss_dict["mlp_reg"]
                        },
                        step=epoch+nn_epoch
                    )
                    partial_results_df = partial_results_df.assign(
                        mlp_reg=reg_loss_dict["mlp_reg"]
                    )
                if conf.output_reg > 0:
                    wandb.log(
                        {
                            "output_regularisation_loss": reg_loss_dict["nn_contrib_reg"],
                        },
                        step=epoch+nn_epoch
                    )
                    partial_results_df = partial_results_df.assign(
                        nn_contrib_reg=reg_loss_dict["nn_contrib_reg"]
                    )

                print(
                    f"epoch {nn_epoch} | "
                    f"loss_train {loss_train} | "
                    f"nn_contrib_reg {reg_loss_dict['nn_contrib_reg'] if reg_loss_dict is not None else 'n/a'} | "
                    f"discontinuity_penalty {reg_loss_dict['discontinuity_penalty'] if reg_loss_dict is not None else 'n/a'} | "
                    f"mlp_reg {reg_loss_dict['mlp_reg'] if reg_loss_dict is not None else 'n/a'} | "
                    f"loss_val {loss_val} | "
                )

                # Check early stopping criterion
                if early_stopping_params.use_early_stopping:
                    # Update early stopper
                    early_stopper = early_stopper.update(loss_val)
                    # Debugging statements
                    print(
                        f"has improved? {early_stopper.has_improved} | "
                        f"patience count {early_stopper.patience_count}"
                    )
                    # Log current patience count
                    wandb.log(
                        {
                            "patience_counter": early_stopper.patience_count,
                        },
                        step=epoch+nn_epoch,
                    )
                    partial_results_df = partial_results_df.assign(ES_patience_counter=early_stopper.patience_count)

                    # Stop training if we have run out of patience
                    if early_stopper.should_stop:
                        print(f'Met early stopping criteria, breaking at epoch {epoch+nn_epoch}')
                        break
            elif nn_epoch == n_epochs - 1:
                # Log final loss values for ADAM training stage
                wandb.log(
                    {
                        "final_loss_train_ADAM": loss_train,
                        "final_loss_val_ADAM": loss_val,
                    },
                    step=epoch+nn_epoch,
                )

            # Append epoch results to results list with added conf details
            results_dfs.append(partial_results_df.assign(**conf.__dict__))

        # Update epoch
        epoch += nn_epoch

        # Plot fit (train/val) at end of training -- using best model on validation dataset
        log_fit_plots(
            model=best_model, **plot_args,
            conf=conf,
            colors=colors, epoch=epoch,
            plot_name="optax", plot_subdir="optax",
        )

        # Log final values of mechanistic parameters + final epoch value (can vary if using early stopping)
        wandb.log(
            {
                **{
                    f"final_{param}_ADAM": transform_dynamic_params(model.func.dynamic_parameters, conf)[i]
                    for i, param in enumerate(dynamic_param_labels[conf.dynamical_model])
                },
                "final_epoch": epoch,
            },
            step=epoch
        )

        # Skip optimistix altogether if NN training failed due to too many invalid losses
        if metric_handler.should_break:
            wandb.finish()
            return best_model

        # Update epoch
        epoch += 1
        # Serialise the best ADAM model
        best_model.save(model_save_dir / (conf.__str__() + "_best_ADAM.eqx"))

    # Train with BFGS via optimistix
    if max_iter_optx_finetune > 0:  # otherwise end training and return trained model
        # Setup filter spec
        if conf.freeze_mechanistic_params_optx:
            bfgs_filter_spec = filter_spec_freeze_mechanistic
        elif conf.freeze_nn_optx:
            bfgs_filter_spec = filter_spec_freeze_nn
        else:
            bfgs_filter_spec = eqx.is_array

        wandb_log = True
        if conf.second_opt == "optx":
            model, sol, static, _, loss_val, epoch, partial_results_df = train_optimistix(
                model=model,
                filter_spec=bfgs_filter_spec,
                conf=conf,
                optx_config=optx_config,
                base_loss=loss_dict["base_loss_fn"],
                loss_optx=loss_dict["loss_optx"],
                val_base_loss=loss_dict["val_base_loss_fn"],
                val_loss_optx=loss_dict["val_loss_optx"],
                epoch=epoch,
                num_epochs=max_iter_optx_finetune,
                u0_train=u0,
                u0_val=u0_val,
                t_save_train=t_save_train,
                t_span_train=t_span_train,
                training_data=training_data,
                t_save_val=t_save_val,
                t_span_val=t_span_val,
                validation_data=validation_data,
                frozen_mechanistic=conf.freeze_mechanistic_params_optx,
                frozen_nn=conf.freeze_nn_optx,
                zero_out_mlp=False,
                wandb_log=True,
                wandb_log_final=True,
            )
        else:
            model, sol, static, _, loss_val, epoch, partial_results_df = train_fides(
                model=model,
                filter_spec=bfgs_filter_spec,
                conf=conf,
                loss_grad_fn=loss_dict["loss_grad_fn"],
                epoch=epoch,
                max_iter=max_iter_optx_finetune,
                u0_train=u0,
                u0_val=u0_val,
                t_save_train=t_save_train,
                t_span_train=t_span_train,
                training_data=training_data,
                t_save_val=t_save_val,
                t_span_val=t_span_val,
                validation_data=validation_data,
                loss_fn=loss_dict["loss_fn"],
                val_loss_fn=loss_dict["val_loss_fn"],
                zero_out_mlp=False,
                wandb_log_final=wandb_log,
            )

        if wandb_log: # if not, partial_results will be None
            results_dfs.append(partial_results_df.assign(**conf.__dict__))

        # Update best_model
        if loss_val < loss_val_min:
            best_model = model

        # Plot eigenvalue and singular value spectrum
        # hessian = get_hessian(sol.value, static, (u0, t_save_train, t_span_train, training_data), conf)
        # hessian_metrics, eigs, svs = get_hessian_metrics(hessian, return_eigs_svs=True)
        # eig_sv_plot(
        #     eigs, svs,
        #     conf,
        #     epoch+sol.stats["num_steps"],
        # )

        # Log plots again after further training with optimistix/BFGS
        log_fit_plots(
            model=model, **plot_args,
            conf=conf,
            colors=colors, epoch=epoch+sol.stats["num_steps"],
            plot_name=conf.second_opt,
            plot_subdir=conf.second_opt,
        )

    # Save results dataframe to CSV
    results_total = pd.concat(results_dfs)
    results_total.to_csv(
        results_dir /
        conf.dynamical_model /
        (conf.__str__() + "_results.csv")
    )

    # Get W&B run directory
    # wandb_stripped_dir = wandb.run.dir.rsplit('/files', 1)[0]
    # command = f"wandb sync {wandb_stripped_dir}"
    wandb.finish()
    # TODO check whether we are using online/offline mode on cluster!
    # # Log everything to W&B after end of run
    # try:
    #     _ = subprocess.run(command, shell=True)
    # except subprocess.CalledProcessError as e:
    #     raise ValueError(f"Error syncing wandb directory: {e}")
    # Serialise best model overall
    best_model.save(model_save_dir / (conf.__str__() + "_best_final.eqx"))
    if not conf.serialise_curriculum_models:
        # Return best_model on validation dataset
        return best_model
    else:
        for cl_ind, epoch in enumerate(model_save_steps_start):
            models_curriculum_stage_start[epoch].save(
                model_save_dir / (conf.__str__() + f"_ADAM_CL{cl_ind}_start.eqx")
            )
        for cl_ind, epoch in enumerate(model_save_steps_end):
            models_curriculum_stage_end[epoch].save(
                model_save_dir / (conf.__str__() + f"_ADAM_CL{cl_ind}_end.eqx")
            )
        return best_model, {**models_curriculum_stage_start, **models_curriculum_stage_end}
