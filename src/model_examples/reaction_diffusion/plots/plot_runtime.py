"""Per-epoch runtime against state-space size, one line per method; panels = variant.

Reads mean_adam_epoch_time, which excludes the compile epoch. Treat it as a relative comparison
between methods rather than a wall-clock predictor -- for multiple shooting it has been observed
to overestimate the realised job duration.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.model_examples.reaction_diffusion.plots.rd_plot_common import (
    METHODS, VARIANTS, add_common_args, load_runs, resolve, save, state_dim,
)

USECOLS = ["mean_adam_epoch_time", "shooting_mode", "curriculum_learning", "multistart_job"]


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    experiment, outdir = resolve(args)
    grids = args.grids

    data = {(v, g): load_runs(experiment, v, g, USECOLS) for v in VARIANTS for g in grids}

    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(12, 5), sharey=True)
    for ax, variant in zip(np.atleast_1d(axes), VARIANTS):
        for label, shooting_mode, cl, cmap_name in METHODS:
            colour = plt.get_cmap(cmap_name)(0.72)
            xs, ys = [], []
            for grid in grids:
                df = data[(variant, grid)]
                if df is None:
                    continue
                sub = df[(df.shooting_mode == shooting_mode) & (df.cl == cl)]
                # one value per run; first() because it is constant within a run
                vals = sub.groupby("multistart_job").mean_adam_epoch_time.first().dropna()
                if len(vals):
                    xs.append(state_dim(grid))
                    ys.append(float(vals.median()))
            if xs:
                order = np.argsort(xs)
                ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], "o-", color=colour,
                        lw=2.2, ms=8, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([state_dim(g) for g in grids])
        ax.set_xticklabels([f"{g}x{g}\n({state_dim(g)})" for g in grids])
        ax.set_title(variant.upper(), fontsize=13)
        ax.set_xlabel("grid  (state dim)", fontsize=10)
        ax.grid(True, which="both", ls=":", alpha=0.35)
        if variant == VARIANTS[0]:
            ax.set_ylabel("per-epoch time  (s, median over seeds)", fontsize=10)
            ax.legend(fontsize=10, title="method")

    fig.suptitle(f"Per-epoch runtime vs state-space size, by method  ({experiment})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, outdir, "rd_runtime_by_grid", args.dpi, not args.no_pdf)


if __name__ == "__main__":
    main()
