import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path
from typing import Optional


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
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
):
    plot_func = sns.boxplot if kind == "box" else sns.barplot

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
        if xscale is not None:
            plt.xscale(xscale)
        if yscale is not None:
            plt.yscale(yscale)
        if ylim is not None:
            plt.ylim(ylim)
        plt.tight_layout()
        if save_plots:
            plt.savefig(fig_dir / f"{plot_name}_{shooting_mode}.pdf", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
