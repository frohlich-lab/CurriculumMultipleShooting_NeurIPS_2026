"""Final extrapolation MSE and mechanistic-parameter recovery against state-space size.

Top row : final extrapolation (validation) MSE per method; panels = variant.
Bottom  : diffusion-parameter recovery, UDE only -- the NODE variant has no mechanistic
          parameters (they are frozen to zero). The grey dashed line is the median error of the
          initial perturbed guess (epoch 0), so it is visible whether training improved on its
          starting point at all.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.model_examples.reaction_diffusion.plots.rd_plot_common import (
    METHODS, VARIANTS, add_common_args, final_per_seed, load_runs, param_error, resolve, save,
    state_dim,
)

USECOLS = ["epoch", "mse_val", "a", "b", "shooting_mode", "curriculum_learning", "multistart_job"]


def jitter(x, index, n):
    """Spread methods slightly on a log x axis so overlapping seed dots stay legible."""
    return x * (1 + 0.04 * (index - (n - 1) / 2))


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    experiment, outdir = resolve(args)
    grids = args.grids

    data = {(v, g): load_runs(experiment, v, g, USECOLS) for v in VARIANTS for g in grids}
    fig, axes = plt.subplots(2, 2, figsize=(13, 10),
                            gridspec_kw={"hspace": 0.33, "wspace": 0.20})

    def style_grid_axis(ax, title, ylabel=None):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([state_dim(g) for g in grids])
        ax.set_xticklabels([f"{g}x{g}\n({state_dim(g)})" for g in grids])
        ax.grid(True, which="both", ls=":", alpha=0.35)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("grid  (state dim)", fontsize=10)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=10)

    # --- final extrapolation MSE ---
    for j, variant in enumerate(VARIANTS):
        ax = axes[0, j]
        for m, (label, shooting_mode, cl, cmap_name) in enumerate(METHODS):
            colour = plt.get_cmap(cmap_name)(0.72)
            xs, ys = [], []
            for grid in grids:
                df = data[(variant, grid)]
                if df is None:
                    continue
                fin = final_per_seed(df, (shooting_mode, cl))
                if fin.empty:
                    continue
                x = state_dim(grid)
                ax.scatter(np.full(len(fin), jitter(x, m, len(METHODS))), fin.mse_val.values,
                           s=20, color=colour, alpha=0.40, edgecolors="none", zorder=3)
                xs.append(x)
                ys.append(float(fin.mse_val.median()))
            if xs:
                ax.plot(xs, ys, "o-", color=colour, lw=2.2, ms=7, zorder=4, label=label)
        style_grid_axis(ax, f"{variant.upper()}   final extrapolation MSE",
                        "final extrap (val) MSE\nline = median, dots = seeds" if j == 0 else None)
        if j == 0:
            ax.legend(fontsize=9.5, title="method")

    # --- UDE parameter recovery ---
    ax = axes[1, 0]
    initial = {}
    for m, (label, shooting_mode, cl, cmap_name) in enumerate(METHODS):
        colour = plt.get_cmap(cmap_name)(0.72)
        xs, ys = [], []
        for grid in grids:
            df = data[("ude", grid)]
            if df is None:
                continue
            sub = df[(df.shooting_mode == shooting_mode) & (df.cl == cl)]
            if sub.empty:
                continue
            fin = sub.groupby("multistart_job").tail(1)
            first = sub.groupby("multistart_job").head(1)      # epoch 0 = the perturbed guess
            initial.setdefault(grid, []).extend(param_error(first.a, first.b))
            err = param_error(fin.a, fin.b)
            x = state_dim(grid)
            ax.scatter(np.full(len(err), jitter(x, m, len(METHODS))), err,
                       s=20, color=colour, alpha=0.40, edgecolors="none", zorder=3)
            xs.append(x)
            ys.append(float(np.median(err)))
        if xs:
            ax.plot(xs, ys, "o-", color=colour, lw=2.2, ms=7, zorder=4, label=label)
    if initial:
        gx = sorted(initial)
        ax.plot([state_dim(g) for g in gx], [float(np.median(initial[g])) for g in gx],
                "--", color="0.45", lw=2.0, zorder=2, label="initial guess")
    style_grid_axis(ax, "UDE   diffusion-parameter recovery  (a, b)",
                    "mean |log10(est/true)|\norders of magnitude off (0 = exact)")
    ax.legend(fontsize=9.5)

    axes[1, 1].axis("off")
    axes[1, 1].text(0.5, 0.5,
                    "NODE variant has no mechanistic\nparameters to recover\n\n"
                    "(the network learns the full RHS and the\ndynamic parameters are frozen to "
                    "zero, so\nrecovery is defined for the hybrid UDE only)",
                    ha="center", va="center", fontsize=11, color="0.35", linespacing=1.6)

    fig.suptitle("Extrapolation accuracy and parameter recovery vs state-space size  "
                 f"({experiment})", fontsize=13.5, y=0.97)
    save(fig, outdir, "rd_extrapolation_and_recovery", args.dpi, not args.no_pdf)


if __name__ == "__main__":
    main()
