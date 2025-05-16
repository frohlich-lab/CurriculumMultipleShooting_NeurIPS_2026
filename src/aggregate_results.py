import numpy as np
import os
import pandas as pd

from generate_run_configs import generate_param_combinations
from src.common import dynamic_param_labels, TRAINING_RESULTS, MERGED_TRAINING_RESULTS, plots_dir, true_dynamic_parameters
from src.model_training.training_configuration import DYNAMICAL_MODEL, EXPERIMENT_NAME, N_EPOCHS, N_STARTS
from src.utils import *



dfs = {dynamical_model: [] for dynamical_model in DYNAMICAL_MODEL}

hyperparam_groups = [
    "mlp_layer_width", "mlp_depth",
    "init_scale",
    "use_bias", "use_final_bias",
    "activation_fn_name",
    "optimiser", "max_lrate", "lrate_span", "lrate_decay", "warmup_fct", "opt_steps", "opt_mult",
    "use_early_stop",
    "mlp_reg_exp", "mlp_reg_strength", "output_reg",
    "shooting_mode", "shooting_window_length", "shooting_iv_init",
    "multiple_shooting_penalty",
    "curriculum_learning", "num_curriculum_steps", "max_curriculum_epochs",
    "dynamic_params_scale", "multistart_deviation",
    "dynamic_params_perturb_method",
    "node_augmentation",
    "second_opt",
    "normalise_data", "normalise_timepoints",
    "training_setting", "mlpexp_target",
]
metrics = [
    "loss_train", "loss_val", "max_grad_norm",
]
log_metrics = [
    "condition_number",
    "frobenius_norm_diagonal", "frobenius_norm_off_diagonal",
    "ratio_frobenius_norm_offdiag_to_diag",
    "max_abs_diagonal_element", "max_abs_off_diagonal_element",
    "max_eigenvalue", "min_eigenvalue",
    "max_sv", "min_sv",
]

for hyperparam_config in generate_param_combinations(N_STARTS):
    training_result_file = TRAINING_RESULTS.format(**hyperparam_config, experiment_name=EXPERIMENT_NAME)
    if os.path.exists(training_result_file):
        df = pd.read_csv(training_result_file)
        dfs[hyperparam_config["dynamical_model"]].append(df)
del df

for dynamical_model in DYNAMICAL_MODEL:
    if len(dfs[dynamical_model]) == 0:
        print(f"No results for {dynamical_model}")
        continue
    else:
        # Add dynamical model specific parameters/metrics
        metrics_model = metrics + dynamic_param_labels[dynamical_model]

        # Clean up dataframe
        results_df = clean_results(
            df=pd.concat(dfs[dynamical_model]),
            attributes_to_fix=["mlpexp_target", "shooting_iv_init", "discontinuity_penalty"]
        )

        # Get dataframe with start and final ADAM epoch losses
        start_epoch_df, final_epoch_df = compute_epoch_subset(
            df=results_df,
            metrics=metrics_model,
            group_cols=hyperparam_groups,
            max_epochs=N_EPOCHS
        )

        # Assign and check figure directory
        fig_dir = plots_dir / dynamical_model
        fig_dir.mkdir(exist_ok=True, parents=True)

        for loss_col in ["loss_train", "loss_val"]:
            # Boxplot of training/validation loss across settings
            tiled_plot(
                df=final_epoch_df,
                row_col="output_reg",
                col_col="multiple_shooting_penalty",
                kind="box",
                x_col="multistart_deviation",
                y_col=loss_col,
                hue_col="curriculum_learning",
                fig_dir=plots_dir / "lotka_volterra",
                plot_name=f"final_{loss_col}_boxplot",
                ylim=(1e-5, 1e10),
                yscale="log"
            )


        # Compute the fraction of runs that achieve a final validation loss below a given threshold (here 1e-3)
        fractions = compute_success_fraction(df=final_epoch_df, threshold=1e-3, group_cols=hyperparam_groups)
        tiled_plot(
            df=fractions,
            row_col="output_reg",
            col_col="multiple_shooting_penalty",
            kind="bar",
            x_col="multistart_deviation",
            y_col="fraction_below_0.001",
            hue_col="curriculum_learning",
            fig_dir=fig_dir,
            plot_name="fraction_below_0.001",
            ylim=(0, 1)
        )

        # Combine info from start and final epochs
        merged = start_epoch_df[hyperparam_groups + ["multistart_job", "loss_train", "loss_val"]].merge(
            final_epoch_df[hyperparam_groups + ["multistart_job", "loss_train", "loss_val"]],
            on=hyperparam_groups + ["multistart_job"],
            how="left",
        ).rename(
            columns={
                "loss_val_x": "loss_val_starting_epoch",
                "loss_val_y": "loss_val_final_epoch",
                "loss_train_x": "loss_train_starting_epoch",
                "loss_train_y": "loss_train_final_epoch",
            }
        ).reset_index()
        # Handle infinite and missing values
        merged.replace([np.inf, -np.inf], np.nan, inplace=True)
        merged.dropna(subset=[
            "loss_val_starting_epoch", "loss_val_final_epoch",
            "loss_train_starting_epoch", "loss_train_final_epoch"
        ], inplace=True)
        # Exclude absurdly large values -- TODO need to refine threshold here!
        merged = merged[(merged.loss_val_starting_epoch < 1e10) & (merged.loss_val_final_epoch < 1e10)]

        # Correlation analysis: loss_val at starting and final ADAM epochs; final ADAM loss_train and loss_val
        summary_df = compute_loss_correlations(merged, hyperparam_groups)
        for y_col in ["corr_start_vs_final", "corr_train_vs_val"]:
            tiled_plot(
                df=summary_df,
                row_col="output_reg",
                col_col="multiple_shooting_penalty",
                kind="bar",
                x_col="multistart_deviation",
                y_col=y_col,
                hue_col="curriculum_learning",
                fig_dir=fig_dir,
                plot_name=y_col,
                ylim=(-1, 1)
            )

        # For Lotka-Volterra (ground truth dynamic params available), compute the fraction of runs
        # that infer parameters within a given tolerance of the true ones
        if dynamical_model == "lotka_volterra":
            tolerance = 0.2  # ±20%
            tolerance_params = compute_fraction_params_within_tolerance(
                final_epoch_df,
                true_dynamic_parameters["lotka_volterra"],
                hyperparam_groups=hyperparam_groups,
                tolerance=tolerance
            )
            tiled_plot(
                df=tolerance_params,
                row_col="output_reg",
                col_col="multiple_shooting_penalty",
                kind="bar",
                x_col="multistart_deviation",
                y_col="fraction_within_tolerance",
                hue_col="curriculum_learning",
                fig_dir=plots_dir / "lotka_volterra",
                plot_name=f"fraction_within_tolerance{tolerance}",
            )


        # # Group by all hyperparams in hyperparam_groups and compute mean and std profile for all metrics across each epoch
        # grouped = final_df.groupby(hyperparam_groups + ["epoch"])[metrics].agg(['mean', 'std'])
        #
        # # Flatten MultiIndex columns (optional but recommended for clarity)
        # grouped.columns = [
        #     f"{metric}_{stat}" for metric, stat in grouped.columns
        # ]
        # grouped = grouped.reset_index()
        # log_final = final_df.dropna(subset=log_metrics)
        # log_grouped = log_final.groupby(hyperparam_groups + ["epoch"])[log_metrics].agg(['mean', 'std'])
        # # Flatten MultiIndex columns (optional but recommended for clarity)
        # log_grouped.columns = [
        #     f"{metric}_{stat}" for metric, stat in log_grouped.columns
        # ]
        # log_grouped = log_grouped.reset_index()
        # # Merge the two DataFrames on the common columns
        # grouped = pd.merge(grouped, log_grouped, on=hyperparam_groups + ["epoch"], how='left')
        # results_df.to_csv(
        #     str(MERGED_TRAINING_RESULTS).format(
        #         dynamical_model=dynamical_model,
        #         experiment_name=EXPERIMENT_NAME
        #     ), index=False
        # )
