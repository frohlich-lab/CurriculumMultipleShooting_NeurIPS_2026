import itertools as itt
import pandas as pd
import plotnine as p9

from pathlib import Path
from typing import Optional


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
    for activation_fn_name, max_lrate, multiple_shooting_penalty in itt.product(
        sorted(df.activation_fn_name.unique()),
        sorted(df.max_lrate.unique()),
        sorted(df.multiple_shooting_penalty.unique())
    ):
        data = df[
            (df.activation_fn_name == activation_fn_name) &
            (df.max_lrate == max_lrate) &
            (df.multiple_shooting_penalty == multiple_shooting_penalty)
        ].copy()

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

            aes_base = p9.aes(**aes_kwargs)

            p = (
                p9.ggplot(summary, aes_base)
                + p9.geom_ribbon(
                    p9.aes(ymin='min_y', ymax='max_y', fill=hue_col),
                    alpha=0.1,
                    colour="None"
                )
                + p9.geom_line()
                + p9.facet_grid(f'{row_col} ~ {col_col}')
                + p9.labs(
                    x=x_col,
                    y=f"Mean {y_col}, min-max range",
                    color=hue_col,
                    fill=hue_col,
                    linetype=style_col if style_col else None
                )
                + p9.theme(figure_size=(12, 8))
            )
            if ref_line_y is not None:
                p += p9.geom_hline(yintercept=ref_line_y, linetype='dashed', color='black')

        else:
            # Expand base aesthetics using raw data (y_col) for bar/box plots
            aes_kwargs["y"] = y_col
            if kind == "bar":
                aes_kwargs["fill"] = hue_col
            aes_mapping = p9.aes(**aes_kwargs)

            if kind == "box":
                geom_obj = p9.geom_boxplot()
            elif kind == "bar":
                geom_obj = p9.geom_bar(stat='identity', position='dodge')
            else:
                raise ValueError(f"Invalid plot kind: {kind}. Use 'box', 'bar', or 'line'.")

            p = (
                p9.ggplot(data, aes_mapping)
                + geom_obj
                + p9.facet_grid(f'{row_col} ~ {col_col}')
                + p9.labs(
                    x=x_col,
                    y=y_col,
                    color=hue_col,
                )
                + p9.theme(figure_size=(12, 8))
            )
            if (kind == "box") and (ref_line_y is not None):
                p += p9.geom_hline(yintercept=ref_line_y, linetype='dashed', color='black')

        # Standard color assignment
        color_dict = {
            "bfgs": "violet",
            "single": "firebrick",
            "single_bfgs": "darkviolet",
            "single_CL": "darkorange",
            "multiple": "darkgreen",
            "multiple_CL": "springgreen",
        }

        # Only add if hue_var is known and values exist
        unique_hue_vals = data[hue_col].unique()
        valid_colors = {k: v for k, v in color_dict.items() if k in unique_hue_vals}
        if valid_colors:
            p += p9.scale_color_manual(values=valid_colors)
            if kind in ["bar", "line"]:
                p += p9.scale_fill_manual(values=valid_colors)

        # Apply axis scalign and coordinate zooming if specified
        if yscale == "log":
            p += p9.scale_y_log10()
        if y_limits is not None:
            p += p9.coord_cartesian(ylim=y_limits)

        if xscale == "log":
            p += p9.scale_x_log10()
        if x_limits is not None:
            p += p9.coord_cartesian(xlim=x_limits)

        # Save or show
        if save_plots:
            path = fig_dir / f"{plot_name}__{activation_fn_name}__{max_lrate}__{multiple_shooting_penalty}.{plot_fmt}"
            p.save(str(path), dpi=300, verbose=False)
        else:
            p.show()
