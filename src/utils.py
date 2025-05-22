import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from src.common import TRAINING_RESULTS, true_dynamic_parameters
from src.generate_run_configs import generate_param_combinations
from plotnine import *
from pathlib import Path
from typing import Optional, Union


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


def tiled_plot(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    kind: str,  # box / bar
    x_col: str,
    y_col: str,
    hue_col: str,
    fig_dir: Path,
    plot_name: str,
    save_plots: bool = True,
    ylim: Optional[tuple] = None,
    xlim: Optional[tuple] = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
):
    if kind == "box":
        plot_func = sns.boxplot
    elif kind == "bar":
        plot_func = sns.barplot
    elif kind == "line":
        plot_func = sns.lineplot
    else:
        raise ValueError(f"Invalid plot kind: {kind}. Use 'box', 'bar', or 'line'.")

    for shooting_mode in sorted(df.shooting_mode.unique()):
        g = sns.FacetGrid(
            data=df[df.shooting_mode == shooting_mode],
            row=row_col,
            row_order=sorted(df[row_col].unique()),
            col=col_col,
            col_order=sorted(df[col_col].unique()),
            margin_titles=True,
            height=4,
            aspect=1.2
        )

        g.map_dataframe(
            plot_func,
            x=x_col,
            y=y_col,
            hue=hue_col,
            palette="tab10",
        )

        # Extract legend from the first non-empty axis
        for ax in g.axes.flat:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                g._legend = ax.legend(
                    handles, labels, title=hue_col,
                    loc="upper right", bbox_to_anchor=(1.05, 1)
                )
                break

        g.set_titles(col_template=f"{col_col}" + ": {col_name}", row_template=f"{row_col}" + ": {row_name}")
        g.set_axis_labels(x_col, y_col)
        g.figure.suptitle(f"Shooting Mode: {shooting_mode}", y=1.02, fontsize=12)

        # Adjust axis limits and scales
        if ylim is not None:
            plt.ylim(ylim)
        if xlim is not None:
            plt.xlim(xlim)

        if xscale is not None:
            plt.xscale(xscale)
        if yscale is not None:
            plt.yscale(yscale)

        plt.tight_layout()
        if save_plots:
            plt.savefig(fig_dir / f"{plot_name}_{shooting_mode}.pdf", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def tiled_plot_plotnine(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    kind: str,
    x_col: str,
    y_col: str,
    hue_col: str,
    fig_dir: Path,
    plot_name: str,
    save_plots: bool = True,
    ref_line_y: Optional[float] = None,
    y_limits: Optional[tuple] = None,
    x_limits: Optional[tuple] = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    style_col: Optional[str] = None,
    plot_fmt: Optional[str] = "png",
):
    for shooting_mode in sorted(df.shooting_mode.unique()):
        data = df[df.shooting_mode == shooting_mode].copy()

        # Preprocessing
        if kind == "line":
            for col in [x_col, y_col]:
                # Convert to numeric if not already
                data[col] = pd.to_numeric(data[col], errors='coerce')
        else:
            # For bar/box plots, x_col needs to be str/categorical
            data[x_col] = data[x_col].astype(str)

        required_cols = [x_col, y_col, hue_col, row_col, col_col]
        if style_col is not None:
            required_cols.append(style_col)
        data = data.dropna(subset=required_cols)

        # Convert hue_col and style_col to string
        data[hue_col] = data[hue_col].astype(str)
        if style_col is not None:
            data[style_col] = data[style_col].astype(str)

        # Define base aesthetics
        aes_kwargs = {
            'x': x_col,
            'color': hue_col
        }

        if kind == "line":
            # Grouping keys
            group_cols = [row_col, col_col, x_col, hue_col]
            if style_col:
                group_cols.append(style_col)

            # Perform aggregation: mean (line), min, max (shaded region, i.e. ribbon)
            summary = (
                data.groupby(group_cols)
                    .agg(mean_y=(y_col, 'mean'), min_y=(y_col, 'min'), max_y=(y_col, 'max'))
                    .reset_index()
            )

            # Construct group label/ID
            if style_col:
                summary['group_label'] = summary[hue_col] + "_" + summary[style_col]
            else:
                summary['group_label'] = summary[hue_col]

            # Expand base aesthetics
            aes_kwargs["y"] = 'mean_y'
            aes_kwargs["group"] = 'group_label'
            if style_col:
                aes_kwargs['linetype'] = style_col

            aes_base = aes(**aes_kwargs)

            p = (
                ggplot(summary, aes_base)
                + geom_ribbon(
                    aes(ymin='min_y', ymax='max_y', fill=hue_col),
                    alpha=0.1,
                    colour="None"
                )
                + geom_line()
                + facet_grid(f'{row_col} ~ {col_col}')
                + labs(
                    title=f"Shooting Mode: {shooting_mode}",
                    x=x_col,
                    y=f"Mean {y_col}, min-max range",
                    color=hue_col,
                    fill=hue_col,
                    linetype=style_col if style_col else None
                )
                + theme(figure_size=(12, 8))
            )
            if ref_line_y is not None:
                p += geom_hline(yintercept=ref_line_y, linetype='dashed', color='black')

        else:
            # Expand base aesthetics using raw data (y_col) for bar/box plots
            aes_kwargs["y"] = y_col
            if kind == "bar":
                aes_kwargs["fill"] = hue_col
            aes_mapping = aes(**aes_kwargs)

            if kind == "box":
                geom_obj = geom_boxplot()
            elif kind == "bar":
                geom_obj = geom_bar(stat='identity', position='dodge')
            else:
                raise ValueError(f"Invalid plot kind: {kind}. Use 'box', 'bar', or 'line'.")

            p = (
                ggplot(data, aes_mapping)
                + geom_obj
                + facet_grid(f'{row_col} ~ {col_col}')
                + labs(
                    title=f"Shooting Mode: {shooting_mode}",
                    x=x_col,
                    y=y_col,
                    color=hue_col,
                )
                + theme(figure_size=(12, 8))
            )
            if (kind == "box") and (ref_line_y is not None):
                p += geom_hline(yintercept=ref_line_y, linetype='dashed', color='black')

        # Apply axis scalign and coordinate zooming if specified
        if yscale == "log":
            p += scale_y_log10()
        if y_limits is not None:
            p += coord_cartesian(ylim=y_limits)

        if xscale == "log":
            p += scale_x_log10()
        if x_limits is not None:
            p += coord_cartesian(xlim=x_limits)

        # Save or show
        if save_plots:
            path = fig_dir / f"{plot_name}_{shooting_mode}.{plot_fmt}"
            p.save(str(path), dpi=300, verbose=False)
        else:
            p.show()


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
                        for dynamic_param in true_dynamic_parameters[hyperparam_config["dynamical_model"]]:
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
