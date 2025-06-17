import numpy as np
# import pandas as pd

from src.common import dynamic_param_labels, plots_dir, true_dynamic_parameters
from src.model_training.training_configuration import DYNAMICAL_MODEL, EXPERIMENT_NAME, N_EPOCHS_ADAM, N_STARTS
from src.utils import *


hyperparam_groups = [
    "mlp_layer_width", "mlp_depth",
    "init_scale",
    "use_bias", "use_final_bias",
    "activation_fn_name",
    "optimiser", "max_lrate", "lrate_span", "lrate_decay", "warmup_fct", "opt_steps", "opt_mult", "opt_sched_type",
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
expanded_hyperparam_groups = hyperparam_groups + ["multistart_job"]

# Base training metrics
metrics = [
    "loss_train", "loss_val", "max_grad_norm",
]
# Hessian-derived metrics
log_metrics = [
    # "condition_number",
    "hessian_trace",
    "frobenius_norm_diagonal", "frobenius_norm_off_diagonal",
    "ratio_frobenius_norm_offdiag_to_diag",
    "max_abs_diagonal_element", "max_abs_off_diagonal_element",
    "max_eigenvalue",
    # "min_eigenvalue",
    "max_sv",
    # "min_sv",
]

# Define experiment-specific plotting directory
fig_dir = plots_dir / EXPERIMENT_NAME

# Assemble result dataframe dictionary from training output CSVs
df_dict = fetch_result_df_dict(
    dynamical_models=DYNAMICAL_MODEL,
    num_starts=N_STARTS,
    experiment_name=EXPERIMENT_NAME,
    attributes_to_fix=["mlpexp_target", "shooting_iv_init", "discontinuity_penalty"],
    filter_results=True,
    loss_train_threshold=1e10,
)


for dynamical_model in DYNAMICAL_MODEL:
    if df_dict[dynamical_model] is None:
        print(f"No results for {dynamical_model}")
        continue
    else:
        results_df = df_dict[dynamical_model]
        # Ensure missing values are represented as the string "None"
        for col in ["optimiser", "second_opt", "mlpexp_target"]:
            if col in results_df.columns:
                results_df[col] = results_df[col].fillna("None").astype(str)
        # Add dynamical model specific parameters/metrics
        metrics_model = metrics + dynamic_param_labels[dynamical_model]

        # Instantiate the model-specific plotting subdirectory
        fig_subdir = fig_dir / dynamical_model
        fig_subdir.mkdir(exist_ok=True, parents=True)

        # Exclude runs with validation loss exceeding group-wise 1.5*interquartile range
        results_df_filtered = exclude_loss_val_outside_iqr(
            df=results_df,
            hyperparam_groups=hyperparam_groups,
        )
        del results_df

        def classify_strategy(row):
            if row["shooting_mode"] != "single":
                return "other"  # or skip entirely
            if row["optimiser"] == "None" and row["second_opt"] != "None":
                return "pure_bfgs"
            elif row["curriculum_learning"]:
                return "single+curriculum"
            else:
                return "single"

        results_df_filtered["strategy"] = results_df_filtered.apply(classify_strategy, axis=1)
        strategy_order = ["pure_bfgs", "single", "single+bfgs", "single+curriculum", "other"]

        results_df_filtered["strategy"] = pd.Categorical(
            results_df_filtered["strategy"],
            categories=strategy_order,
            ordered=True
        )

        # Get dataframe with start and final ADAM epoch losses
        start_epoch_df, adam_final_epoch_df, bfgs_final_epoch_df = compute_epoch_subset(
            df=results_df_filtered,
            metrics=metrics_model,
            group_cols=hyperparam_groups,
        )

        final_epoch_df = pd.concat(
            [df for df in [adam_final_epoch_df, bfgs_final_epoch_df] if df is not None],
            axis=0
        )

        # After computing final_epoch_df via concat
        final_epoch_df["strategy_phase"] = final_epoch_df["strategy"]


        # Override only for post-ADAM BFGS runs
        final_epoch_df.loc[
            (final_epoch_df["curriculum_stage"] == -1)
            & (final_epoch_df["optimiser"] == "adam")
            & (final_epoch_df["curriculum_learning"] == False),
            "strategy_phase"
        ] = "single+bfgs"

        final_epoch_df["strategy"] = final_epoch_df["strategy_phase"]
        final_epoch_df.drop(columns=["strategy_phase"], inplace=True)
        final_epoch_df["strategy"] = pd.Categorical(
            final_epoch_df["strategy"],
            categories=strategy_order,
            ordered=True
        )
        final_epoch_df = final_epoch_df.reset_index().drop(columns=["index"])

        col_param = "multiple_shooting_penalty" if final_epoch_df.multiple_shooting_penalty.nunique() > 1 \
            else "shooting_window_length"

        # Boxplot of final model metrics across settings
        for metric in metrics_model:
            if metric == "alpha":
                ref_val = true_dynamic_parameters["lotka_volterra"].tolist()[0]
            elif metric == "delta":
                ref_val = true_dynamic_parameters["lotka_volterra"].tolist()[1]
            else:
                ref_val = None
            tiled_plot_plotnine(
                df=final_epoch_df,
                row_col="output_reg",
                col_col=col_param,
                kind="box",
                x_col="multistart_deviation",
                y_col=metric,
                hue_col={"single": "strategy", "multiple": "curriculum_learning"},
                fig_dir=fig_subdir,
                plot_name=f"final_{metric}_boxplot",
                y_limits=(-5, 10) if metric in ["loss_train", "loss_val"] else None,  # with plotnine, y_limits for log-scaled axes must be specified in log-space
                yscale="log",
                ref_line_y=ref_val,
            )


        # Plot average metric profiles across settings
        log_epochs_adam = results_df_filtered[
            results_df_filtered[log_metrics].notna().any(axis=1)
        ].epoch.unique()

        log_epochs_bfgs = results_df_filtered[
            results_df_filtered["epoch"] > 3999
            ].epoch.unique()
        log_epochs = np.sort(np.unique(np.concatenate([log_epochs_adam, log_epochs_bfgs])))

        results_df_reduced = results_df_filtered[results_df_filtered.epoch.isin(log_epochs)].copy()
        loss_df_reduced = results_df_reduced[
            hyperparam_groups + ["multistart_job", "epoch", "strategy"] + metrics_model
        ].copy()
        hessian_df = results_df_reduced[
            hyperparam_groups + ["multistart_job", "epoch", "strategy"] + log_metrics
        ].copy()
        hessian_df = hessian_df[hessian_df.epoch.isin(log_epochs_adam)]
        del results_df_reduced

        # Boxplot of hessian metrics across settings (final recorded log-epoch)
        for log_metric in log_metrics:
            tiled_plot_plotnine(
                df=hessian_df[hessian_df.epoch == hessian_df.epoch.max()],
                row_col="output_reg",
                col_col=col_param,
                kind="box",
                x_col="multistart_deviation",
                y_col=log_metric,
                hue_col={"single": "strategy", "multiple": "curriculum_learning"},
                fig_dir=fig_subdir,
                plot_name=f"final_{log_metric}_boxplot",
                # y_limits=(-5, 10),  # with plotnine, y_limits for log-scaled axes must be specified in log-space
                yscale="log",
            )


        # Loss metrics + max_grad_norm
        for model_metric in metrics_model:
            if model_metric == "alpha":
                ref_val = true_dynamic_parameters["lotka_volterra"].tolist()[0]
            elif model_metric == "delta":
                ref_val = true_dynamic_parameters["lotka_volterra"].tolist()[1]
            elif model_metric in ["loss_train", "loss_val"]:
                ref_val = 1e-2
            else:
                ref_val = None

            tiled_plot_plotnine(
                df=loss_df_reduced,
                row_col='output_reg',
                col_col=col_param,
                kind='line',
                x_col='epoch',
                y_col=model_metric,
                hue_col={"single": "strategy", "multiple": "curriculum_learning"},
                style_col='multistart_deviation',
                yscale='log',
                xscale='log',
                y_limits=(-5, 15) if model_metric in ["loss_train", "loss_val"] else None,
                fig_dir=fig_subdir,
                plot_name=f"{model_metric}_profile",
                ref_line_y=ref_val,
            )

        # Hessian-derived metrics
        for log_metric in log_metrics:
            tiled_plot_plotnine(
                df=hessian_df,
                row_col='output_reg',
                col_col=col_param,
                kind='line',
                x_col='epoch',
                y_col=log_metric,
                hue_col={"single": "strategy", "multiple": "curriculum_learning"},
                style_col='multistart_deviation',
                yscale='log',
                xscale='log',
                fig_dir=fig_subdir,
                plot_name=f"{log_metric}_profile",
            )


        # Compute the fraction of runs that achieve a final validation loss below a given threshold (here 1e-2)
        fractions = compute_success_fraction(
            df=final_epoch_df,
            threshold=2e-2,
            group_cols=hyperparam_groups + ["strategy"]
        )
        tiled_plot_plotnine(
            df=fractions,
            row_col="output_reg",
            col_col=col_param,
            kind="bar",
            x_col="multistart_deviation",
            y_col="fraction_below_0.02",
            hue_col={"single": "strategy", "multiple": "curriculum_learning"},
            fig_dir=fig_subdir,
            plot_name="fraction_below_0.02",
            y_limits=(0, 1)
        )

        # Match multistart jobs with known 'single+bfgs' strategy
        single_bfgs_jobs = final_epoch_df[
            final_epoch_df["strategy"] == "single+bfgs"
            ][hyperparam_groups + ["multistart_job"]].drop_duplicates()

        start_epoch_df = start_epoch_df.merge(
            single_bfgs_jobs.assign(strategy_phase="single+bfgs"),
            on=hyperparam_groups + ["multistart_job"],
            how="left"
        )

        # Use assigned value if present, fallback to original
        start_epoch_df["strategy"] = start_epoch_df["strategy_phase"].fillna(start_epoch_df["strategy"])
        start_epoch_df.drop(columns=["strategy_phase"], inplace=True)

        # Re-cast to categorical
        start_epoch_df["strategy"] = pd.Categorical(
            start_epoch_df["strategy"],
            categories=strategy_order,
            ordered=True
        )

        # Combine info from start and final epochs
        merged = start_epoch_df[
            hyperparam_groups + ["multistart_job", "strategy", "loss_train", "loss_val"]
        ].merge(
            final_epoch_df[
                hyperparam_groups + ["multistart_job", "strategy", "loss_train", "loss_val"]
            ],
            on=hyperparam_groups + ["multistart_job", "strategy"],
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
        # # Exclude absurdly large values -- this should be taken care of by the IQR filtering above
        # merged = merged[(merged.loss_val_starting_epoch < 1e10) & (merged.loss_val_final_epoch < 1e10)]

        # Correlation analysis: loss_val at starting and final ADAM epochs; final ADAM loss_train and loss_val
        summary_df = compute_loss_correlations(merged, hyperparam_groups + ["strategy"])
        for y_col in ["corr_start_vs_final", "corr_train_vs_val"]:
            tiled_plot_plotnine(
                df=summary_df,
                row_col="output_reg",
                col_col=col_param,
                kind="bar",
                x_col="multistart_deviation",
                y_col=y_col,
                hue_col={"single": "strategy", "multiple": "curriculum_learning"},
                fig_dir=fig_subdir,
                plot_name=y_col,
                y_limits=(-1, 1)
            )

        # For Lotka-Volterra (ground truth dynamic params available), compute the fraction of runs
        # that infer parameters within a given tolerance of the true ones
        if dynamical_model == "lotka_volterra":
            tolerance = 0.2  # ±20%
            tolerance_params = compute_fraction_params_within_tolerance(
                final_epoch_df,
                true_dynamic_parameters["lotka_volterra"],
                hyperparam_groups=hyperparam_groups + ["strategy"],
                tolerance=tolerance
            )
            tiled_plot_plotnine(
                df=tolerance_params,
                row_col="output_reg",
                col_col=col_param,
                kind="bar",
                x_col="multistart_deviation",
                y_col="fraction_within_tolerance",
                hue_col={"single": "strategy", "multiple": "curriculum_learning"},
                fig_dir=fig_subdir,
                plot_name=f"fraction_within_tolerance{tolerance}",
            )
