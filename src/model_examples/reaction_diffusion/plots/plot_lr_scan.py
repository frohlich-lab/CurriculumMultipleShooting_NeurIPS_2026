"""Learning-rate scan: final training MSE against learning rate, per method and variant.

Selection is made on TRAINING MSE (a learning rate should be chosen on optimisation quality, not
on held-out accuracy). Individual seeds are drawn, because the medians alone hide that the good
configurations are heavy-tailed across seeds.

A learning rate that was run in a different experiment (e.g. reused from the main sweep) can be
folded in with --extra-experiment, so the shared value is not re-run.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.model_examples.reaction_diffusion.plots.rd_plot_common import (
    METHODS, VARIANTS, add_common_args, load_runs, resolve, save,
)

USECOLS = ["epoch", "mse_train", "shooting_mode", "curriculum_learning", "multistart_job",
           "max_lrate"]


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--grid", type=int, default=8,
                       help="grid size the scan was run on")
    parser.add_argument("--extra-experiment", default=None,
                       help="second experiment to merge in (for learning rates reused from "
                            "another sweep)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                       help="restrict to these multistart seeds (default: all present)")
    args = parser.parse_args()
    experiment, outdir = resolve(args)

    def combined(variant):
        frames = [df for exp in (experiment, args.extra_experiment) if exp
                  for df in [load_runs(exp, variant, args.grid, USECOLS)] if df is not None]
        if not frames:
            return None
        df = pd.concat(frames, ignore_index=True)
        if args.seeds is not None:
            df = df[df.multistart_job.isin(args.seeds)]
        return df

    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(13, 5.5), sharey=True)
    for ax, variant in zip(np.atleast_1d(axes), VARIANTS):
        df = combined(variant)
        for m, (label, shooting_mode, cl, cmap_name) in enumerate(METHODS):
            colour = plt.get_cmap(cmap_name)(0.72)
            offset = 1 + 0.05 * (m - (len(METHODS) - 1) / 2)   # fan methods apart on the log axis
            xs, ys = [], []
            if df is None:
                continue
            sub = df[(df.shooting_mode == shooting_mode) & (df.cl == cl)]
            for lrate, group in sub.groupby("max_lrate"):
                fin = group.sort_values("epoch").groupby("multistart_job").tail(1)
                if fin.empty:
                    continue
                ax.scatter(np.full(len(fin), lrate * offset), fin.mse_train.values,
                           s=26, color=colour, alpha=0.5, edgecolors="none", zorder=3)
                xs.append(float(lrate))
                ys.append(float(fin.mse_train.median()))
            if xs:
                order = np.argsort(xs)
                xs, ys = np.asarray(xs)[order], np.asarray(ys)[order]
                ax.plot(xs * offset, ys, "o-", color=colour, lw=2.2, ms=8, zorder=4, label=label)
                best = int(np.argmin(ys))
                ax.scatter([xs[best] * offset], [ys[best]], s=200, facecolors="none",
                           edgecolors=colour, linewidths=2.2, zorder=5)
                print(f"  {variant.upper():<4} {label:<3} best lr={xs[best]:g} "
                      f"({ys[best]:.3e})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls=":", alpha=0.35)
        ax.set_xlabel("max learning rate", fontsize=11)
        ax.set_title(variant.upper(), fontsize=13)
        if variant == VARIANTS[0]:
            ax.set_ylabel("final training MSE\nline = median, dots = seeds", fontsize=11)
            ax.legend(fontsize=11, title="method")

    fig.suptitle(f"Learning-rate scan on the {args.grid}x{args.grid} grid - final training MSE"
                 "   (open ring = per-method best)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, outdir, f"rd_lr_scan_{args.grid}x{args.grid}", args.dpi, not args.no_pdf)


if __name__ == "__main__":
    main()
