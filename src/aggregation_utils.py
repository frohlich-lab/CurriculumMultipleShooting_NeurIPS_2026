import numpy as np
import os
import pandas as pd

from src.common import TRAINING_RESULTS, dynamic_param_labels
from src.generate_run_configs import generate_param_combinations
from typing import Union


def clean_results(df: pd.DataFrame, attributes_to_fix: list[str]):
    fill_value_dict = {
        attribute: "None"
        for attribute in attributes_to_fix
    }
    df.fillna(fill_value_dict, inplace=True)
    df.drop(columns=[col for col in df.columns if "Unnamed" in col], inplace=True, errors='ignore')
    df.reset_index(drop=True, inplace=True)
    return df


def compute_epoch_subset(
        df: pd.DataFrame,
        metrics: list[str],
        group_cols: list[str],
        max_epochs: int
):
    # Subset to model metrics only
    loss_subset = df[["epoch", "multistart_job"] + group_cols + metrics]
    # First epoch and last ADAM epoch (< max_epochs)
    first_epoch = loss_subset["epoch"].min()
    last_epoch = loss_subset[loss_subset["epoch"] < max_epochs]["epoch"].max()
    return loss_subset[loss_subset["epoch"] == first_epoch], loss_subset[loss_subset["epoch"] == last_epoch]


def compute_success_fraction(df: pd.DataFrame, threshold: float, group_cols: list[str]):
    df["success"] = df["loss_val"] < threshold
    return (
        df.groupby(group_cols)["success"]
        .agg(["mean", "count", "sum"])
        .rename(columns={"mean": f"fraction_below_{threshold}", "count": "num_runs", "sum": "num_successful"})
        .reset_index()
    )

def compute_fraction_params_within_tolerance(
    final_epoch: pd.DataFrame,
    true_params: tuple,
    hyperparam_groups: list,
    tolerance: float = 0.1
) -> pd.DataFrame:
    rows = []

    for group_key, group in final_epoch.groupby(hyperparam_groups):
        # Extract inferred parameters from the group
        inferred_params = group[["alpha", "delta"]].to_numpy()

        # Check if inferred params are within tolerance for each run
        lower_bound = np.array(true_params) * (1 - tolerance)
        upper_bound = np.array(true_params) * (1 + tolerance)

        # Boolean mask: each row must be within bounds for both alpha and delta
        is_within = np.all((inferred_params >= lower_bound) & (inferred_params <= upper_bound), axis=1)
        fraction_within = is_within.mean()

        # Store result
        result = {col: val for col, val in zip(hyperparam_groups, group_key)} if isinstance(group_key, tuple) else {hyperparam_groups[0]: group_key}
        result.update({
            "fraction_within_tolerance": fraction_within,
            "num_runs": len(group),
            "num_within": is_within.sum()
        })
        rows.append(result)

    return pd.DataFrame(rows)


def compute_loss_correlations(merged_df: pd.DataFrame, group_cols: list[str]):
    rows = []
    for combo, group in merged_df.groupby(group_cols):
        result = dict(zip(group_cols, combo))
        result["corr_start_vs_final"] = (
            group[["loss_val_starting_epoch", "loss_val_final_epoch"]].corr().iloc[0, 1]
            if group.loss_val_starting_epoch.nunique() > 1 and group.loss_val_final_epoch.nunique() > 1
            else np.nan
        )
        result["corr_train_vs_val"] = (
            group[["loss_val_final_epoch", "loss_train_final_epoch"]].corr().iloc[0, 1]
            if group.loss_train_final_epoch.nunique() > 1 and group.loss_val_final_epoch.nunique() > 1
            else np.nan
        )
        rows.append(result)
    return pd.DataFrame(rows)


def fetch_result_df_dict(
        dynamical_models: Union[list, set],
        num_starts: int,
        experiment_name: str,
        attributes_to_fix: list[str],
        filter_results: bool = False,
        loss_train_threshold: float = 1e10,
) -> dict[str, pd.DataFrame]:
    """
    Fetches the result dataframes for each dynamical model and concatenates them. It optionally filters out poorly
    performing runs based on initial and final training loss. Finally, it cleans the dataframes by removing
    unnecessary columns and filling NaN values in optional attributes.

    :param dynamical_models: list/set of dynamical models to fetch results for.
    :param num_starts: number of random starts for each hyperparameter configuration.
    :param experiment_name: name of the experiment to fetch results for.
    :param attributes_to_fix: list of attributes to fill NaN values for.
    :param filter_results: whether to filter out poorly performing runs based on initial and final training loss.
    :param loss_train_threshold: threshold for filtering poorly performing runs.

    :return: dictionary with dynamical models (str) as keys and concatenated dataframes as values.
    """
    dfs = {dynamical_model: [] for dynamical_model in dynamical_models}
    df_dict = {}
    for hyperparam_config in generate_param_combinations(num_starts):
        training_result_file = TRAINING_RESULTS.format(**hyperparam_config, experiment_name=experiment_name)
        if os.path.exists(training_result_file):
            df = pd.read_csv(training_result_file)
            if filter_results:
                # Discard bad runs according to initial and final training loss
                if (
                    (df[df.epoch == df.epoch.min()]["loss_train"].values < loss_train_threshold)
                    and
                    (df[df.epoch == df.epoch.max()]["loss_train"].values < loss_train_threshold)
                ):
                    cleaned_df = clean_results(
                        df=df,
                        attributes_to_fix=attributes_to_fix,
                    )
                    # If timepoints have been normalised, rescale dynamic parameters
                    if hyperparam_config["normalise_timepoints"]:
                        for dynamic_param in dynamic_param_labels[hyperparam_config["dynamical_model"]]:
                            if (dynamic_param in cleaned_df.columns) and ("time_rescaling" in cleaned_df.columns):
                                cleaned_df[dynamic_param] /= cleaned_df["time_rescaling"]

                    dfs[hyperparam_config["dynamical_model"]].append(
                        cleaned_df
                    )
                else:
                    continue
            else:
                dfs[hyperparam_config["dynamical_model"]].append(
                    clean_results(
                        df=df,
                        attributes_to_fix=attributes_to_fix,
                    )
                )
    for dynamical_model in dynamical_models:
        if len(dfs[dynamical_model]) > 0:
            df_dict[dynamical_model] = pd.concat(dfs[dynamical_model])
        else:
            df_dict[dynamical_model] = None
    return df_dict


def exclude_loss_val_outside_iqr(
        df: pd.DataFrame,
        hyperparam_groups: list[str],
        iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    # Step 1: get loss_val at the starting epoch
    start_vals = df[df.epoch == df.epoch.min()]

    # Step 2: compute and apply IQR filtering mask per group
    def iqr_filter(group):
        q1 = group['loss_val'].quantile(0.25)
        q3 = group['loss_val'].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        # Determine if the group's starting value is within IQR bounds
        return group[(group['loss_val'] >= lower) & (group['loss_val'] <= upper)]

    filtered_start_vals = start_vals.groupby(hyperparam_groups, group_keys=False).apply(iqr_filter)

    # Step 3: get valid hyperparam combinations to keep
    valid_combinations = filtered_start_vals[hyperparam_groups + ["multistart_job"]].drop_duplicates()

    # Step 4: filter the full dataset using valid hyperparam combinations
    filtered_df = df.merge(valid_combinations, on=hyperparam_groups + ["multistart_job"], how='inner')
    return filtered_df
