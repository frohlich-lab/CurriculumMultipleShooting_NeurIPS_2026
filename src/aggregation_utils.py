import h5py
import numpy as np
import os
import pandas as pd

from pathlib import Path
from src.common import TRAINING_RESULTS, dynamic_param_labels
from src.generate_run_configs import generate_param_combinations
from typing import Tuple, Union


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
        group_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Extracts:
      - First epoch (start of training, ADAM or BFGS)
      - Final ADAM epoch (if optimiser used)
      - Final BFGS/second optimiser epoch (if second optimiser used)

    Parameters:
        df (pd.DataFrame): Full results dataframe.
        metrics (list[str]): List of metric column names.
        group_cols (list[str]): Grouping keys for unique runs (e.g., hyperparameter settings).

    Returns:
        (start_epoch_df, adam_final_epoch_df or None, bfgs_final_epoch_df or None)
    """
    cols = ["epoch", "multistart_job", "curriculum_stage"] + group_cols + metrics
    df = df[cols].copy()
    group_keys = group_cols + ["multistart_job"]

    # 1. First epoch of training (ADAM or BFGS)
    start_epoch_df = df.groupby(group_keys).apply(
        lambda g: g[g.epoch == g.epoch.min()]
    ).reset_index(drop=True)

    # 2. Final ADAM epoch (if any rows with curriculum_stage != -1)
    if (df.curriculum_stage != -1).any():
        adam_df = df[df.curriculum_stage != -1]
        adam_final_epoch_df = adam_df.groupby(group_keys).apply(
            lambda g: g[g.epoch == g.epoch.max()]
        ).reset_index(drop=True)
    else:
        adam_final_epoch_df = None

    # 3. Final BFGS epoch (if any rows with curriculum_stage == -1)
    if (df.curriculum_stage == -1).any():
        bfgs_df = df[df.curriculum_stage == -1]
        bfgs_final_epoch_df = bfgs_df.groupby(group_keys).apply(
            lambda g: g[g.epoch == g.epoch.max()]
        ).reset_index(drop=True)
    else:
        bfgs_final_epoch_df = None

    return start_epoch_df, adam_final_epoch_df, bfgs_final_epoch_df


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
                    # # If timepoints have been normalised, rescale dynamic parameters -> moved to training
                    # if hyperparam_config["normalise_timepoints"]:
                    #     for dynamic_param in dynamic_param_labels[hyperparam_config["dynamical_model"]]:
                    #         if (dynamic_param in cleaned_df.columns) and ("time_rescaling" in cleaned_df.columns):
                    #             cleaned_df[dynamic_param] /= cleaned_df["time_rescaling"]

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
    group_keys = hyperparam_groups + ["multistart_job"]

    # Step 1: Get first logged epoch per run
    start_vals = df.groupby(group_keys, group_keys=False).apply(
        lambda g: g[g.epoch == g.epoch.min()]
    ).reset_index(drop=True)

    # Step 2: Apply IQR filter on starting loss_val, per hyperparameter group
    def iqr_filter(group):
        q1 = group["loss_val"].quantile(0.25)
        q3 = group["loss_val"].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        return group[(group["loss_val"] >= lower) & (group["loss_val"] <= upper)]

    filtered_start_vals = start_vals.groupby(hyperparam_groups, group_keys=False).apply(iqr_filter)

    # Step 3: Keep full run data for those runs
    valid_combinations = filtered_start_vals[group_keys].drop_duplicates()
    filtered_df = df.merge(valid_combinations, on=group_keys, how='inner')

    return filtered_df


def write_results_hdf5(
    results_df: pd.DataFrame,
    out_path: Path,
    scan_attributes: list[str],
    deviation_col: str = "multistart_deviation",
    job_col: str = "multistart_job",
    epoch_col: str = "epoch",
) -> Path:
    """
    Writes results into HDF5 with hierarchy:
        /<scan_attr>/<scan_value>/<multistart_deviation>/job_<id>/arrays/{epoch, metric1, metric2, ...}
    """

    def _compute_central_values(df: pd.DataFrame, scan_attributes: list[str]) -> dict[str, object]:
        """
        Determine the central value of each scan attribute as the most frequent (mode).
        Falls back to median for numeric columns if mode is not defined.
        """
        central = {}
        for attr in scan_attributes:
            col = df[attr]
            try:
                mode_val = col.mode(dropna=True).iloc[0]  # central values = most frequent for that attribute
                central[attr] = mode_val
            except Exception:
                # fallback to median if no mode (e.g. continuous numeric)
                if pd.api.types.is_numeric_dtype(col):
                    central[attr] = float(col.median())
                else:
                    central[attr] = col.dropna().iloc[0] if not col.dropna().empty else np.nan
        return central

    def _split_constants_global(df: pd.DataFrame, cols: list[str]) -> tuple[dict, list[str]]:
        """Find columns that are constant across the ENTIRE df."""
        consts, varying = {}, []
        for c in cols:
            if c not in df.columns:
                continue
            n_unique = df[c].nunique(dropna=False)
            if n_unique <= 1:
                consts[c] = df[c].iloc[0] if n_unique == 1 else np.nan
            else:
                varying.append(c)
        return consts, varying

    exclude = {epoch_col, job_col, deviation_col} | set(scan_attributes)
    metrics = [
        c for c in results_df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(results_df[c])
    ]

    out_path = Path(out_path)
    if out_path.exists():
        out_path.unlink()

    with h5py.File(out_path, "w") as h5:
        meta = h5.require_group("meta")
        meta.attrs["metrics_requested"] = np.array(metrics, dtype=h5py.string_dtype())
        meta.attrs["scan_attributes"] = np.array(scan_attributes, dtype=h5py.string_dtype())
        meta.attrs["job_col"] = job_col
        meta.attrs["epoch_col"] = epoch_col
        meta.attrs["deviation_col"] = deviation_col

        # Get globally constant (not scanned) parameters
        g_consts, base_metrics = _split_constants_global(results_df, metrics)
        # Save constants once at /meta as hp_* and record which final metrics will be recorded
        for k, v in g_consts.items():
            meta.attrs[f"hp_{k}"] = v if not isinstance(v, str) else np.array(v, dtype=h5py.string_dtype())
        meta.attrs["metrics_written"] = np.array(base_metrics, dtype=h5py.string_dtype())

        # Get central values
        central_values = _compute_central_values(results_df, scan_attributes)

        # Build trees for each scan attribute
        for scan_attr in scan_attributes:
            top = h5.require_group(str(scan_attr))

            # Determine other scan attributes (not currently scanned)
            other_attrs = [attr for attr in scan_attributes if attr != scan_attr]

            # Build a master Boolean mask to subset to central values for all other attributes
            mask = np.logical_and.reduce([
                (results_df[attr] == central_values[attr]) for attr in other_attrs
            ])

            # Iterate across values for the given scan hyperparameter
            for val in results_df[scan_attr].drop_duplicates().tolist():
                # Refine master mask by subsetting to value at hand
                attr_mask = mask & (results_df[scan_attr] == val)
                df_val = results_df[attr_mask].copy()

                val_grp = top.require_group(str(val))
                # Preserve raw value
                try:
                    val_grp.attrs["value_raw"] = val
                except TypeError:
                    val_grp.attrs["value_raw"] = str(val)

                if deviation_col not in df_val.columns:
                    raise KeyError(f"Expected '{deviation_col}' column in results.")

                # Group by multistart_deviation
                for dev in df_val[deviation_col].drop_duplicates().tolist():
                    df_dev = df_val[df_val[deviation_col] == dev].copy()
                    dev_grp = val_grp.require_group(f"multistart_deviation={dev}")
                    # Preserve raw deviation
                    try:
                        dev_grp.attrs["deviation_raw"] = dev
                    except TypeError:
                        dev_grp.attrs["deviation_raw"] = str(dev)

                    if job_col not in df_dev.columns:
                        raise KeyError(f"Expected '{job_col}' column in results.")

                    for job_id in sorted(df_dev[job_col].dropna().unique().tolist()):
                        df_job = df_dev[df_dev[job_col] == job_id].copy()
                        job_grp = dev_grp.require_group(f"job_{int(job_id)}")

                        if epoch_col not in df_job.columns:
                            raise KeyError(f"Expected '{epoch_col}' column in results.")

                        # Deduplicate epochs (keep last per epoch)
                        df_job_dedup = df_job.sort_values(epoch_col).drop_duplicates(epoch_col, keep="last")

                        # Arrays only: epochs + globally-varying metrics (constants removed)
                        arr_grp = job_grp.require_group("arrays")
                        job_sorted = df_job_dedup.sort_values(epoch_col)

                        # epoch vector
                        epochs = pd.to_numeric(job_sorted[epoch_col], errors="coerce").astype("Int64").to_numpy()
                        if "epoch" in arr_grp:
                            del arr_grp["epoch"]
                        arr_grp.create_dataset("epoch", data=epochs)

                        # metrics vectors (only those that vary globally)
                        for m in base_metrics:
                            series = pd.to_numeric(job_sorted[m], errors="coerce").to_numpy()
                            if m in arr_grp:
                                del arr_grp[m]
                            arr_grp.create_dataset(m, data=series)

    return out_path


def read_metric_trajectory(
    h5_path: str,
    scan_attr: str,
    scan_value: Union[str, int, float],
    multistart_deviation: Union[str, int, float],
    job_id: int,
    metric: str,
    y_only: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Read (epochs, metric) arrays from an HDF5 results file with layout:
      /<scan_attr>/<scan_value>/<multistart_deviation>/job_<id>/arrays/{epoch, <metric>}

    Args
    ----
    h5_path : str
        Path to the HDF5 file.
    scan_attr : str
        Name of the scan attribute (top-level group).
    scan_value : str|int|float
        Value of the scan attribute (group under scan_attr).
    multistart_deviation : str|int|float
        Deviation value (group under scan_value).
    job_id : int
        Multistart job id (expects group 'job_<id>').
    metric : str
        Name of the metric dataset inside the 'arrays' group.
    y_only : bool
        If True, return only the metric array (Y). Otherwise, return (epochs, Y).

    Returns
    -------
    np.ndarray or (np.ndarray, np.ndarray)
        If y_only=True, returns metric array (Y). Otherwise (epochs, Y).
    """
    with h5py.File(h5_path, "r") as h5:
        # /<scan_attr>
        if scan_attr not in h5:
            raise KeyError(f"Scan attribute '{scan_attr}' not found. Available: {list(h5.keys())}")
        g_scan = h5[scan_attr]

        # /<scan_attr>/<scan_value>
        g_val = g_scan[str(scan_value)]

        # /<scan_attr>/<scan_value>/<multistart_deviation>
        g_dev = g_val[f"multistart_deviation={multistart_deviation}"]

        # /.../job_<id>
        job_key = f"job_{int(job_id)}"
        if job_key not in g_dev:
            # Offer a hint if they passed a string like 'job_1'
            alt = [n for n in g_dev.keys() if n.startswith("job_")]
            raise KeyError(f"Job group '{job_key}' not found under {g_dev.name}. Available jobs: {alt}")
        g_job = g_dev[job_key]

        # /.../arrays
        if "arrays" not in g_job:
            raise KeyError(f"Group 'arrays' not found under {g_job.name}.")
        arr = g_job["arrays"]

        if "epoch" not in arr:
            raise KeyError(f"'epoch' dataset not found under {arr.name}.")
        epochs = np.array(arr["epoch"][:])

        if metric not in arr:
            raise KeyError(f"Metric '{metric}' not found under {arr.name}. Available: {list(arr.keys())}")
        y = np.array(arr[metric][:])

    return y if y_only else (epochs, y)
