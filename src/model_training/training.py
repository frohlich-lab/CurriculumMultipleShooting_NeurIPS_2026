import diffrax
import equinox as eqx
import jax.numpy as jnp
import optax
import pandas as pd
import time
import wandb
import warnings

from src.common import (
    Conf,
    dynamic_param_labels,
    hessian_metrics_wandb,
    plots_dir,
    results_dir,
    TrainValArgs
)
from dataclasses import replace
from diffrax import SaveAt
from jaxtyping import Array, Float, PyTree
from pathlib import Path
from src.common import TRAINING_RESULTS, reformat_filename
from src.model_training.losses import get_hessian
from src.model_training.model_util import process_mechanistic_params
from src.model_training.training_util import (
    check_and_progress_curriculum,
    get_save_epochs,
    get_train_val_mse,
    generate_log_epochs,
    get_filter_spec,
    get_opt_opt_state_params,
    get_scheduler,
    initialise_wandb,
    make_grads_finite,
    max_grad_norm,
    get_hessian_metrics,
    build_hessian_index_mask,
    update_hessian_index_mask_multiple_curriculum,
    MetricHandler,
    augment_u0
)
from src.model_training.optimistix_training import train_optimistix
from src.model_training.fides_training import train_fides
from src.plotting import log_plots
from src.ude_problem import UDEProblem
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# Some FutureWarning with unhashable DynamicJaxprTracer object in the diffeq solve/call
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_fit_dfs(
        prob: UDEProblem,
        u0,
        t_save,
        t_span,
        t_save_windows,
        data: jnp.ndarray,
        shooting_mode: str,
        curriculum_learning: bool,
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
    elif prob.dynamical_model.startswith("sir"):
        df_pred = pd.DataFrame({"time": sol.ts,
                                "Infected": jnp.log(jnp.absolute(sol.ys[:, 1] + sol.ys[:, 3])),
                                "Recovered + dead": jnp.log(jnp.absolute(sol.ys[:, 2])),
                                })
        # Observed data
        infected, dead_recovered = data[:, 0], data[:, 1]
        df_obs = pd.DataFrame({"time": sol.ts,
                               "Infected": jnp.log(infected),
                               "Recovered + dead": jnp.log(jnp.absolute(dead_recovered)),
                               })
    else:
        raise ValueError(f"Undefined dynamical model {prob.dynamical_model}.")
    if (shooting_mode == "single") or (dataset != "train") or curriculum_learning:
        return df_pred, None, df_obs
    else:
        if shooting_mode == "multiple":  # TODO need to fix this for MS and MS+CL
            # TODO @GiacomoFabrini -- need to fix u0/t_save - provided vs internal?
            sol = prob.solve_multiple_shooting(
                diffrax.Tsit5(),
                prob.initial_values,
                t_save_windows,
            )
        else:
            raise ValueError(f"Undefined shooting mode {shooting_mode}.")
        if prob.dynamical_model == "lotka_volterra":
            df_pred_train = pd.concat([
                pd.DataFrame({"time": sol.ts[i],
                              "prey": sol.ys[i, :, 0],
                              "predator": sol.ys[i, :, 1],
                              }).assign(window=i + 1)
                for i in range(sol.ys.shape[0])  # shooting-window dimension
            ])
        elif prob.dynamical_model.startswith("sir"):
            df_pred_train = pd.concat([
                pd.DataFrame({"time": sol.ts[i],
                              "Infected": jnp.log(jnp.absolute(sol.ys[i, :, 1] + sol.ys[i, :, 3])),
                              "Recovered + dead": jnp.log(jnp.absolute(sol.ys[i, :, 2])),
                              }).assign(window=i + 1)
                for i in range(sol.ys.shape[0])  # shooting-window dimension
            ])
        else:
            raise ValueError(f"Undefined dynamical model {prob.dynamical_model}.")
        return df_pred, df_pred_train, df_obs


# PLOTTING HELPER FUNCTION
def log_fit_plots(
        model: UDEProblem,
        u0: jnp.ndarray,
        t_save_train: Union[SaveAt, Array], t_span_train: List, training_data: jnp.ndarray,
        t_save_val: Union[SaveAt, Array], t_span_val: List, validation_data: jnp.ndarray,
        t_save_windows: jnp.ndarray,
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
            model, u0, t_save, t_span, t_save_windows, data, conf.shooting_mode, conf.curriculum_learning, dataset,
        )
        plot_dir = plots_dir / conf.dynamical_model / plot_subdir
        plot_dir.mkdir(exist_ok=True, parents=True)
        log_plots(
            [df_predict, df_predict_train],
            df_obs,
            dataset,
            conf.shooting_mode,
            conf.curriculum_learning,
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
        input_data: Float[Array, "t d"],  # shape = (num_timepoints, augmented_state_dimension)
        t_save_windows: Array,
        masks: Array,
        last_valid_index: int,
        conf: Conf,
        loss_grad_fn: Callable,
        compute_hessian_metrics: bool,
        base_loss_fn: Callable,
        active_idx: Optional[Array],
        rescaling_args: dict[str, float],
        max_steps_diffrax: int,
):
    (loss_value, reg_loss_dict), grads = loss_grad_fn(
        prob_params=params,
        prob_static=static,
        t_save_windows=t_save_windows,
        masks=masks,
        data=input_data,
        last_valid_index=last_valid_index,
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
    if compute_hessian_metrics and (active_idx is not None):
        args = (t_save_windows, masks, input_data, last_valid_index)
        hessian_metrics = get_hessian_metrics(
            get_hessian(params, static, args, conf, base_loss_fn, rescaling_args, max_steps_diffrax),
            return_eigs_svs=False,
            active_idx=active_idx,
            num_dyn_params=params.func.dynamic_parameters.shape[0] if conf.training_setting == "UDE" else None,
        )
    updates, opt_state = opt.update(grads, opt_state, params)
    params = eqx.apply_updates(params, updates)
    return params, opt_state, loss_value, reg_loss_dict, grads, hessian_metrics


# MAIN TRAINING FUNCTION
def train(
        model: UDEProblem,
        train_args: TrainValArgs,
        val_args: TrainValArgs,
        t_save_windows: Array,
        masks: Array,
        curriculum_params: Optional[dict[int, Any]],
        conf: Conf,
        loss_dict: Dict[str, Callable],
        optx_config: Dict,
        n_epochs: int,
        max_iter_optx_finetune: int,
        model_save_dir: Path,
        plot_args: dict,
        rescaling_args: dict,
        max_steps_diffrax: int,
) -> UDEProblem:
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
    :param t_save_windows:
        Array of timepoints to use for training.
    :param masks:
        Array of binary masks to use for training.
    :param curriculum_params: Optional[dict]
        Dictionary of curriculum learning parameters.
    :param conf:
        configuration (hyperparameters).
    :param loss_dict:
        Dictionary of loss functions.
    :param optx_config:
        Configuration for optimistix solver (rtol, atol).
    :param n_epochs:
        Number of training iterations (epochs).
    :param max_iter_optx_finetune:
        Maximum number of iterations for further training via optimistix.
    :param model_save_dir:
        Filepath of directory to save the trained models.
    :param plot_args:
        Arguments for plotting train/val data.
    :param rescaling_args:
        Arguments for rescaling data/timepoints.
    :param max_steps_diffrax:
        Maximum number of iterations for diffrax solve.
    """

    # Check training setting in configuration
    if conf.training_setting not in ["UDE", "NODE"]:
        # if not properly set, set to default value of UDE
        conf = replace(conf, training_setting="UDE")

    WANDB_LOG = False

    if WANDB_LOG:
        # Initialise wandb and define metrics
        initialise_wandb(
            conf=conf,
            model=model,
        )

    # Initialise variables
    epoch, nn_epoch = 0, 0
    best_model, mse_val_min = model, jnp.inf
    model_save_steps_start = []
    overall_wall_t0, overall_process_t0 = time.perf_counter(), time.process_time()
    adam_wall_duration, adam_process_duration = jnp.nan, jnp.nan
    steady_epoch_times_mean, steady_epoch_times_std = jnp.nan, jnp.nan

    if not conf.curriculum_learning:
        if n_epochs > 0:
            num_save_models = 10
            save_epochs = get_save_epochs(n_epochs, num_save_models)
    else:
        # Curriculum save points: start and end of each stage
        model_save_steps_start = [curriculum_params[stage]["start"] for stage in curriculum_params]
        midpoints = [
            curriculum_params[stage]["start"] +
            (curriculum_params[stage]["end"] - curriculum_params[stage]["start"]) // 2
            for stage in curriculum_params
        ]
        model_save_steps_end = [curriculum_params[stage]["end"]-1 for stage in curriculum_params]
        save_epochs = model_save_steps_start + midpoints + model_save_steps_end

    # Unpack training and validation args
    _, training_data, t_save_train, t_span_train = train_args.u0, train_args.data, train_args.t_save, train_args.t_span
    u0_val, validation_data, t_save_val, t_span_val = val_args.u0, val_args.data, val_args.t_save, val_args.t_span

    # Add initial zero as starting condition for the 2-norm of the neural network contribution
    u0_val = augment_u0(u0_val)
    plot_args["u0"] = augment_u0(plot_args["u0"])
    u0_train = plot_args["u0"]

    # Define colours for plots
    if conf.dynamical_model == "lotka_volterra":
        colors = ['red', 'black']
    elif conf.dynamical_model.startswith('sir'):
        colors = ['red', 'black']
    # TODO check pendulum!
    elif conf.dynamical_model == "pendulum":
        colors = ['red', 'black', 'blue', 'purple']
    else:
        raise ValueError(f"Undefined dynamical model {conf.dynamical_model}.")

    # Initialise last_valid_index based on shooting-mode
    if conf.shooting_mode == "multiple":
        num_windows = int(jnp.ceil(len(training_data) / (conf.shooting_window_length - 1)))
        last_valid_index = num_windows
    else:
        last_valid_index = 1

    # Ensure results directory is correctly set up
    results_save_dir = results_dir / str(conf.experiment_name) / conf.dynamical_model / "train"
    results_save_dir.mkdir(exist_ok=True, parents=True)
    results_dfs = []

    # STAGE 1: Train with ADAM/ADAM-W
    if n_epochs > 0:
        cl_stage = 1  # initialise CL stage

        # Partition model into learnable / static parameters depending on configuration
        adam_filter_spec = get_filter_spec(model, conf, optimiser="adam")
        params, static = eqx.partition(model, adam_filter_spec)

        if conf.compute_hessian_metrics:
            active_hessian_idx = build_hessian_index_mask(model, conf, adam_filter_spec)
        else:
            active_hessian_idx = None

        # Generate regularly log-spaced epochs for model stat logging - from DMMs
        log_epochs = generate_log_epochs(n_epochs=n_epochs, num_samples=100, min_dist=5)
        # Add first and final epochs
        log_epochs = jnp.unique(
            jnp.sort(
                jnp.concatenate(
                    [log_epochs, jnp.array([0, n_epochs - 1])]
                )
            )
        )

        # Get schedule
        schedule = get_scheduler(conf=conf, n_epochs=n_epochs)
        # Initialise optimiser and optimiser state
        opt, opt_state, params = get_opt_opt_state_params(conf, params, schedule)

        # Training loop
        metric_handler = MetricHandler(patience=5)
        ivs = []
        epoch_wall_times, epoch_process_times = [], []

        for nn_epoch in range(n_epochs):
            # Keep track of initial values for multiple shooting windows
            if conf.shooting_mode == "multiple":
                ivs.append(params.initial_values)
            # Check curriculum stage update
            if conf.curriculum_learning:
                t_save_windows, masks, cl_stage, advancing_cl_stage = check_and_progress_curriculum(
                    t_save_windows, masks, nn_epoch, cl_stage, model_save_steps_start, curriculum_params
                )

                if advancing_cl_stage:
                    # Update last valid window index
                    last_valid_index = curriculum_params[cl_stage]["last_valid_index"]

                    # Update valid Hessian parameter indices
                    if conf.compute_hessian_metrics and (conf.shooting_mode == "multiple"):
                        # Update valid indices for Hessian metric computation
                        active_hessian_idx = update_hessian_index_mask_multiple_curriculum(
                            model, adam_filter_spec, last_valid_index
                        )

                    # Optionally restart optimiser and opt_state, including lr schedule for new curriculum stage
                    if conf.restart_opt_curriculum:
                        print(f"Restarting optimiser at epoch {nn_epoch} for CL stage {cl_stage}")
                        schedule = get_scheduler(conf=conf, n_epochs=n_epochs)
                        opt, opt_state, params = get_opt_opt_state_params(conf, params, schedule)


            # Optionally save model
            if conf.save_landscape_models and nn_epoch in save_epochs:
                # Save model at the end of the epoch
                model_pca_traj_dir = model_save_dir / "checkpoints"
                model_pca_traj_dir.mkdir(exist_ok=True, parents=True)
                epoch_save_path = (
                    model_pca_traj_dir /
                    (conf.__str__() + f"_ep{nn_epoch}_CL{cl_stage}.eqx")
                )
                print(f"Saving model at epoch {nn_epoch}, CL stage {cl_stage} to {epoch_save_path}")
                inference_model = model.replace_mlp(
                    new_mlp=model.func.mlp,
                    switch_to_inference=True,
                ) if conf.use_dropout else model
                inference_model.save(epoch_save_path)

            wall_t0, process_t0 = time.perf_counter(), time.process_time()

            # Take step
            params, opt_state, loss_train, reg_loss_dict, grads, hessian_metrics = make_step(
                params=params,
                static=static,
                opt=opt,
                opt_state=opt_state,
                input_data=training_data,
                t_save_windows=t_save_windows,
                masks=masks,
                last_valid_index=last_valid_index,
                conf=conf,
                loss_grad_fn=loss_dict["loss_grad_fn"],
                compute_hessian_metrics=(conf.compute_hessian_metrics and (nn_epoch in log_epochs)),
                base_loss_fn=loss_dict["base_loss_fn"],
                active_idx=active_hessian_idx if conf.compute_hessian_metrics else None,
                rescaling_args=rescaling_args,
                max_steps_diffrax=max_steps_diffrax,
            )

            jnp.asarray(loss_train).block_until_ready()

            wall_t1, process_t1 = time.perf_counter(), time.process_time()
            epoch_wall_time = wall_t1 - wall_t0
            epoch_process_time = process_t1 - process_t0
            epoch_wall_times.append(epoch_wall_time)
            epoch_process_times.append(epoch_process_time)

            if metric_handler.check_whether_should_break(loss_train):
                break

            # Recombine dynamic and static into full model
            model = eqx.combine(params, static)

            if nn_epoch in log_epochs:
                # Compute validation loss prior to epoch update
                inference_model = model.replace_mlp(
                    new_mlp=model.func.mlp,
                    switch_to_inference=True,
                ) if conf.use_dropout else model

                # compute train and val MSE
                mse_train, mse_val = get_train_val_mse(
                    model=inference_model,
                    u0_train=u0_train,
                    t_save_train=t_save_train,
                    t_span_train=t_span_train,
                    training_data=training_data,
                    u0_val=u0_val,
                    t_save_val=t_save_val,
                    t_span_val=t_span_val,
                    validation_data=validation_data,
                    loss_dict=loss_dict,
                )

                # inference_params, inference_static = eqx.partition(inference_model, adam_filter_spec)

                # mse_val, _ = loss_dict["val_loss_fn"](
                #     prob_params=inference_params,
                #     prob_static=inference_static,
                #     u0=u0_val,
                #     t_save=t_save_val,
                #     t_span=t_span_val,
                #     data=validation_data,
                # )

                # Keep track of best performing model on validation dataset (and corresponding loss)
                if mse_val < mse_val_min:
                    best_model = inference_model
                    mse_val_min = mse_val

                # Save partial training results
                partial_results_df = pd.DataFrame(
                    [{
                        "optimiser": conf.optimiser,
                        "epoch": epoch + nn_epoch,
                        # Curriculum stage
                        "curriculum_stage": cl_stage,
                        # Losses
                        "loss_train": loss_train,
                        "mse_train": mse_train,
                        "mse_val": mse_val,
                        # Training metrics
                        "learning_rate": schedule(nn_epoch),
                        "num_train_samples": curriculum_params[cl_stage]["length"] if conf.curriculum_learning \
                            else len(training_data),
                        "max_grad_norm": max_grad_norm(grads),
                        # Dynamic parameters: raw and rescaled
                        **{
                            f"raw_{param}": process_mechanistic_params(
                                inference_model.func.dynamic_parameters,
                                inference_model.func.options
                            )[i]
                            for i, param in enumerate(dynamic_param_labels[conf.dynamical_model])
                        },
                        **{
                            param: process_mechanistic_params(
                                inference_model.func.dynamic_parameters,
                                inference_model.func.options
                            )[i]/rescaling_args["time_rescaling"]  # ensure actual parameters are properly rescaled
                            for i, param in enumerate(dynamic_param_labels[conf.dynamical_model])
                        },
                        # Initial values
                        **{
                            f"iv_{i}_{j}": inference_model.initial_values[i][j]
                            for i in range(len(inference_model.initial_values))
                            for j in range(inference_model.func.state_dimension)
                        },
                        "data_rescaling": rescaling_args["data_rescaling"],
                        "time_rescaling": rescaling_args["time_rescaling"],
                        "wall_time": epoch_wall_time,
                        "process_time": epoch_process_time,
                    }]
                )

                if WANDB_LOG:
                    # Log loss_train, mse_train, mse_val, learning rate, param values and grads on log-spaced epochs
                    wandb.log(
                        {
                            "loss_train": loss_train,
                            "mse_train": mse_train,
                            "mse_val": mse_val,
                            "learning_rate": schedule(nn_epoch),
                            "num_train_samples": curriculum_params[cl_stage]["length"] if conf.curriculum_learning \
                                else len(training_data),
                            "max_grad_norm": max_grad_norm(grads),
                        },
                        step=epoch+nn_epoch
                    )

                if conf.compute_hessian_metrics:
                    # Add Hessian metrics to partial results
                    partial_results_df = partial_results_df.assign(**hessian_metrics)

                    if WANDB_LOG:
                        # Log Hessian metrics
                        wandb.log(
                            {
                                **hessian_metrics
                            },
                            step=epoch + nn_epoch
                        )

                # Log discontinuity penalty if applied
                if (conf.multiple_shooting_penalty > 0) and (conf.shooting_mode == "multiple"):
                    partial_results_df = partial_results_df.assign(
                        discontinuity_penalty=reg_loss_dict["discontinuity_penalty"],
                    )
                    if WANDB_LOG:
                        wandb.log(
                            {
                                "discontinuity_penalty": reg_loss_dict["discontinuity_penalty"]
                            },
                            step=epoch + nn_epoch
                        )

                # Log regularisation term if applied
                if conf.mlp_reg_strength > 0:
                    partial_results_df = partial_results_df.assign(
                        mlp_reg=reg_loss_dict["mlp_reg"]
                    )
                    if WANDB_LOG:
                        wandb.log(
                            {
                                "regularisation_loss": reg_loss_dict["mlp_reg"]
                            },
                            step=epoch + nn_epoch
                        )

                if conf.output_reg > 0:
                    partial_results_df = partial_results_df.assign(
                        nn_contrib_reg=reg_loss_dict["nn_contrib_reg"]
                    )
                    if WANDB_LOG:
                        wandb.log(
                            {
                                "output_regularisation_loss": reg_loss_dict["nn_contrib_reg"],
                            },
                            step=epoch + nn_epoch
                        )

                print(
                    f"epoch {nn_epoch} | "
                    f"loss_train {loss_train} | "
                    f"nn_contrib_reg {reg_loss_dict['nn_contrib_reg'] if reg_loss_dict is not None else 'n/a'} | "
                    f"discontinuity_penalty {reg_loss_dict['discontinuity_penalty'] if reg_loss_dict is not None else 'n/a'} | "
                    f"mlp_reg {reg_loss_dict['mlp_reg'] if reg_loss_dict is not None else 'n/a'} | "
                    f"mse_train {mse_train} | "
                    f"mse_val {mse_val} | "
                    f"min_sv {hessian_metrics['min_sv']} | "
                )

                if nn_epoch == n_epochs - 1:
                    # Compute final training loss on single-shooting with known u0
                    final_mse_train, final_mse_val = get_train_val_mse(
                        model=model,
                        u0_train=u0_train,
                        t_save_train=t_save_train,
                        t_span_train=t_span_train,
                        training_data=training_data,
                        u0_val=u0_val,
                        t_save_val=t_save_val,
                        t_span_val=t_span_val,
                        validation_data=validation_data,
                        loss_dict=loss_dict,
                    )
                    partial_results_df["final_mse_train"] = final_mse_train
                    partial_results_df["final_mse_val"] = final_mse_val

                    if WANDB_LOG:
                        wandb.log(
                            {
                                "final_mse_train_ADAM": final_mse_train,
                                "final_mse_val_ADAM": final_mse_val,
                            },
                            step=epoch+nn_epoch,
                        )

                # Append epoch results to results list with added conf details
                results_dfs.append(partial_results_df.assign(**conf.__dict__))

        # Update epoch tally
        epoch += nn_epoch

        if WANDB_LOG:
            # Log final values of mechanistic parameters + final epoch value
            wandb.log(
                {
                    **{
                        f"final_{param}_ADAM": process_mechanistic_params(
                            model.func.dynamic_parameters, model.func.options
                        )[i]
                        for i, param in enumerate(dynamic_param_labels[conf.dynamical_model])
                    },
                    "final_epoch": epoch,
                },
                step=epoch
            )
            # Plot fit (train/val) at end of training -- using best model on validation dataset
            log_fit_plots(
                model=best_model, **plot_args,
                conf=conf,
                colors=colors, epoch=epoch,
                plot_name="optax", plot_subdir="optax",
            )

        overall_wall_t1, overall_process_t1 = time.perf_counter(), time.process_time()
        adam_wall_duration = overall_wall_t1 - overall_wall_t0
        adam_process_duration = overall_process_t1 - overall_process_t0

        # Serialise the best ADAM model
        best_model.save(model_save_dir / (conf.__str__() + "_best_ADAM.eqx"))

        # Compute mean/standard deviation of epoch process times
        if len(epoch_process_times) >= 2:
            steady = epoch_process_times[1:]  # discard epoch 0 (JIT compile)
            steady_epoch_times_mean = float(jnp.mean(jnp.array(steady)))
            steady_epoch_times_std = float(jnp.std(jnp.array(steady)))
            print(steady_epoch_times_mean, steady_epoch_times_std)

        # Break if NN training failed due to too many invalid losses
        if metric_handler.should_break:
            if len(results_dfs) > 0:
                # Save partial results anyway
                results_total = pd.concat(results_dfs).assign(
                    mean_adam_epoch_time=steady_epoch_times_mean,
                    std_adam_epoch_time=steady_epoch_times_std
                )
            else:
                # Save CSV with inf stats
                results_total = pd.DataFrame(
                    [{
                        "optimiser": conf.optimiser,
                        "epoch": epoch,
                        # Curriculum stage
                        "curriculum_stage": cl_stage,
                        # Losses
                        "loss_train": jnp.inf,
                        "mse_train": jnp.inf,
                        "mse_val": jnp.inf,
                        # Training metrics
                        "learning_rate": schedule(nn_epoch),
                        "num_train_samples": curriculum_params[cl_stage]["length"] if conf.curriculum_learning \
                            else len(training_data),
                        # Dynamic parameters
                        **{
                            f"raw_{param}": process_mechanistic_params(
                                best_model.func.dynamic_parameters,
                                best_model.func.options
                            )[i]
                            for i, param in enumerate(dynamic_param_labels[conf.dynamical_model])
                        },
                        **{
                            param: process_mechanistic_params(
                                best_model.func.dynamic_parameters,
                                best_model.func.options
                            )[i] / rescaling_args["time_rescaling"]  # ensure actual parameters are properly rescaled
                            for i, param in enumerate(dynamic_param_labels[conf.dynamical_model])
                        },
                        "data_rescaling": rescaling_args["data_rescaling"],
                        "time_rescaling": rescaling_args["time_rescaling"],
                    }]
                )
            results_total.to_csv(reformat_filename(TRAINING_RESULTS.format_map(conf.__dict__)), index=False)
            if WANDB_LOG:
                wandb.finish()
            return best_model

        # Update epoch prior to BFGS training, if any
        epoch += 1

    # Stage 2 - train with BFGS via optimistix/fides
    if max_iter_optx_finetune > 0:  # otherwise end training and return trained model
        bfgs_filter_spec = get_filter_spec(model, conf, optimiser="bfgs")

        if conf.second_opt == "optx":
            model, sol, static, epoch, partial_results_df = train_optimistix(
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
                u0_val=u0_val,
                t_save_windows=t_save_windows,
                masks=masks,
                training_data=training_data,
                t_save_val=t_save_val,
                t_span_val=t_span_val,
                validation_data=validation_data,
                last_valid_index=last_valid_index,
                rescaling_args=rescaling_args,
                zero_out_mlp=False,
                wandb_log=WANDB_LOG
            )
        else:
            model, sol, static, epoch, partial_results_df = train_fides(
                model=model,
                filter_spec=bfgs_filter_spec,
                conf=conf,
                loss_grad_fn=loss_dict["loss_grad_fn"],
                epoch=epoch,
                max_iter=max_iter_optx_finetune,
                t_save_windows=t_save_windows,
                masks=masks,
                training_data=training_data,
                last_valid_index=last_valid_index,
                loss_fn=loss_dict["loss_fn"],
                rescaling_args=rescaling_args,
                zero_out_mlp=False,
                wandb_log=WANDB_LOG
            )

        # Compute final training and validation losses
        final_mse_train_bfgs, final_mse_val_bfgs = get_train_val_mse(
            model=model,
            u0_train=u0_train,
            t_save_train=t_save_train,
            t_span_train=t_span_train,
            training_data=training_data,
            u0_val=u0_val,
            t_save_val=t_save_val,
            t_span_val=t_span_val,
            validation_data=validation_data,
            loss_dict=loss_dict,
        )
        # Log final losses
        partial_results_df["final_mse_train_bfgs"] = final_mse_train_bfgs
        partial_results_df["final_mse_val_bfgs"] = final_mse_val_bfgs

        # Add rescaling info & curriculum stage
        partial_results_df["data_rescaling"] = rescaling_args["data_rescaling"]
        partial_results_df["time_rescaling"] = rescaling_args["time_rescaling"]
        partial_results_df["curriculum_stage"] = -1  # to distinguish from valid values

        protected = {'optimiser'}
        conf_dict = {k: v for k, v in conf.__dict__.items() if k not in protected}
        results_dfs.append(partial_results_df.assign(**conf_dict))

        if WANDB_LOG:
            # Log plots again after further training with optimistix/BFGS
            log_fit_plots(
                model=model, **plot_args,
                conf=conf,
                colors=colors, epoch=epoch + sol.stats["num_steps"],
                plot_name=conf.second_opt,
                plot_subdir=conf.second_opt,
            )

        # Update best_model
        if final_mse_val_bfgs < mse_val_min:
            best_model = model


    # Save results dataframe to CSV
    results_total = pd.concat(results_dfs).assign(
        mean_adam_epoch_time=steady_epoch_times_mean,
        std_adam_epoch_time=steady_epoch_times_std,
        adam_wall_duration=adam_wall_duration,
        adam_process_duration=adam_process_duration,
    )
    train_results_filename = reformat_filename(TRAINING_RESULTS.format_map(conf.__dict__))
    print(f"Saving run results to {train_results_filename}...")
    results_total.to_csv(train_results_filename, index=False)

    if WANDB_LOG:
        wandb.finish()

    # Serialise best model overall
    best_model.save(model_save_dir / (conf.__str__() + "_best_final.eqx"))
    return best_model
