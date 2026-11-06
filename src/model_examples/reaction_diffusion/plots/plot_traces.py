"""Median optimisation traces per method and grid resolution.

Rows = variant x (interpolation, extrapolation) MSE; columns = training method; grid resolution
is encoded as shade within the method's colour family. Curriculum stage boundaries are read from
the logged `curriculum_stage` column, so they are the schedule that actually ran.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model_examples.reaction_diffusion.plots.rd_plot_common import (
    METHODS, SHADE, VARIANTS, add_common_args, load_runs, resolve, save,
)

USECOLS = ["epoch", "mse_train", "mse_val", "shooting_mode", "curriculum_learning",
           "multistart_job", "curriculum_stage"]
ROWS = [("ude", "mse_train", "UDE  interp MSE"), ("ude", "mse_val", "UDE  extrap MSE"),
        ("node", "mse_train", "NODE  interp MSE"), ("node", "mse_val", "NODE  extrap MSE")]


def stage_starts(data, grids, variant, shooting_mode, cl):
    """First epoch of each curriculum stage (identical across grids and seeds); [] if not a
    curriculum method. The last entry is where the curriculum finishes, i.e. the first epoch
    trained on the full interval / all windows.

    Note the logged epochs are quantised by LOG_EVERY_N_EPOCHS, so these are accurate to that
    interval rather than to the exact switch epoch.
    """
    for grid in grids:
        df = data.get((variant, grid))
        if df is None or "curriculum_stage" not in df:
            continue
        sub = df[(df.shooting_mode == shooting_mode) & (df.cl == cl)]
        if sub.empty:
            continue
        first = sub.groupby(["multistart_job", "curriculum_stage"]).epoch.min().reset_index()
        starts = [float(e) for e in first.groupby("curriculum_stage").epoch.median().sort_index()
                  if e > 0]
        if starts:
            return starts
    return []


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    experiment, outdir = resolve(args)
    grids = args.grids

    data = {(v, g): load_runs(experiment, v, g, USECOLS) for v in VARIANTS for g in grids}
    stages = {(v, label): stage_starts(data, grids, v, sm, cl)
              for v in VARIANTS for label, sm, cl, _ in METHODS if cl}

    fig, axes = plt.subplots(len(ROWS), len(METHODS), figsize=(16, 13),
                             sharex=True, sharey="row")
    for i, (variant, metric, row_label) in enumerate(ROWS):
        for j, (label, shooting_mode, cl, cmap_name) in enumerate(METHODS):
            ax = axes[i, j]
            cmap = plt.get_cmap(cmap_name)
            for grid in grids:
                df = data[(variant, grid)]
                if df is None:
                    continue
                sub = df[(df.shooting_mode == shooting_mode) & (df.cl == cl)]
                if sub.empty:
                    continue
                median = sub.groupby("epoch")[metric].median()
                n = int(sub.multistart_job.nunique())
                ax.plot(median.index, median.values, color=cmap(SHADE.get(grid, 0.7)),
                        lw=1.9, label=f"{grid}x{grid} (n={n})")
            starts = stages.get((variant, label), [])
            for k, epoch in enumerate(starts):
                last = k == len(starts) - 1
                ax.axvline(epoch, color="0.25" if last else "0.65", lw=1.8 if last else 0.9,
                           ls="-" if last else ":", alpha=0.9 if last else 0.7, zorder=1)
            ax.set_yscale("log")
            ax.grid(True, which="both", ls=":", alpha=0.3)
            if i == 0:
                ax.set_title(label + (f"   [{len(starts) + 1} stages]" if starts else ""),
                             fontsize=14)
                ax.legend(fontsize=7.5, loc="upper right")
            if j == 0:
                ax.set_ylabel(row_label, fontsize=11)
            if i == len(ROWS) - 1:
                ax.set_xlabel("epoch", fontsize=10)

    fig.suptitle(f"Optimisation traces - median over seeds  ({experiment})\n"
                 "shade = grid resolution, column = method  |  dotted verticals = curriculum "
                 "stage starts, solid = curriculum end",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    save(fig, outdir, "rd_traces_by_grid", args.dpi, not args.no_pdf)


if __name__ == "__main__":
    main()
