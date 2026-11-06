"""Interaction view of the factorial hyperparameter scan.

The continuity penalty on x, one line per shooting window length, facetted by output_reg. Parallel
lines mean the best penalty is independent of the window count; crossing lines mean the two
interact and have to be chosen jointly. This is the reason the scan is run as a full factorial
rather than one axis at a time.

The bottom row carries the degenerate axes: single shooting has no windows and no penalty, so only
output_reg varies (and, for curriculum learning, the number of stages). output_reg is drawn on a
categorical axis because the NODE variant has it forced to zero, which a log axis cannot place.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.model_examples.reaction_diffusion.plots.rd_plot_common import (
    add_common_args, load_runs, resolve, save,
)

USECOLS = ["epoch", "mse_train", "shooting_mode", "curriculum_learning", "multistart_job",
           "output_reg", "shooting_window_length", "multiple_shooting_penalty",
           "num_curriculum_steps"]
ROWS = [("ude", "UDE  MS", "multiple", False, "Greens"),
        ("ude", "UDE  CMS", "multiple", True, "Oranges"),
        ("node", "NODE  MS", "multiple", False, "Greens"),
        ("node", "NODE  CMS", "multiple", True, "Oranges")]


def finals(sub):
    return None if sub.empty else sub.sort_values("epoch").groupby("multistart_job").tail(1)


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--grid", type=int, default=8, help="grid size the scan was run on")
    args = parser.parse_args()
    experiment, outdir = resolve(args)

    data = {v: load_runs(experiment, v, args.grid, USECOLS) for v in ("ude", "node")}
    present = [df for df in data.values() if df is not None]
    if not present:
        raise SystemExit(f"no results found for {experiment} at {args.grid}x{args.grid}")
    windows = sorted({int(w) for df in present for w in df.shooting_window_length.unique() if w > 0})
    penalties = sorted({float(p) for df in present
                        for p in df.multiple_shooting_penalty.unique() if p > 0})
    oreg_levels = {v: sorted({float(o) for o in df.output_reg.unique()}) if df is not None else []
                   for v, df in data.items()}
    ncol = max(len(v) for v in oreg_levels.values()) or 1
    styles = {w: ("o-" if k == 0 else "s--", 0.45 + 0.4 * k / max(len(windows) - 1, 1))
              for k, w in enumerate(windows)}

    fig, axes = plt.subplots(len(ROWS) + 1, ncol, figsize=(4.6 * ncol, 16), sharey=True,
                            squeeze=False, gridspec_kw={"hspace": 0.42, "wspace": 0.22})

    for i, (variant, row_label, shooting_mode, cl, cmap_name) in enumerate(ROWS):
        df = data[variant]
        cmap = plt.get_cmap(cmap_name)
        levels = oreg_levels[variant]
        for j in range(ncol):
            ax = axes[i][j]
            if j >= len(levels):
                ax.axis("off")
                continue
            output_reg = levels[j]
            for window in windows:
                style, shade = styles[window]
                colour = cmap(shade)
                xs, ys = [], []
                for penalty in penalties:
                    if df is None:
                        continue
                    sub = df[(df.shooting_mode == shooting_mode) & (df.cl == cl)
                             & (df.shooting_window_length == window)
                             & np.isclose(df.output_reg, output_reg)
                             & np.isclose(df.multiple_shooting_penalty, penalty)]
                    fin = finals(sub)
                    if fin is None:
                        continue
                    nudge = 1 + 0.06 * (windows.index(window) - (len(windows) - 1) / 2)
                    ax.scatter(np.full(len(fin), penalty * nudge), fin.mse_train.values,
                               s=22, color=colour, alpha=0.45, edgecolors="none", zorder=3)
                    xs.append(penalty)
                    ys.append(float(fin.mse_train.median()))
                if xs:
                    ax.plot(xs, ys, style, color=colour, lw=2.2, ms=7, zorder=4,
                            label=f"window {window}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(penalties)
            ax.set_xticklabels([f"{p:g}" for p in penalties])
            ax.grid(True, which="both", ls=":", alpha=0.35)
            ax.set_xlabel(r"$\lambda_{cont}$", fontsize=10)
            ax.set_title(f"{row_label}   output_reg={output_reg:g}", fontsize=10.5)
            if j == 0:
                ax.set_ylabel("final training MSE", fontsize=10)
                ax.legend(fontsize=8.5)

    # bottom row: the degenerate axes. output_reg on a categorical axis so a forced 0 is placeable.
    ticks = sorted({o for levels in oreg_levels.values() for o in levels})
    index = {o: k for k, o in enumerate(ticks)}

    def positions(values, shift=0.0):
        return np.array([index.get(float(v), np.nan) for v in values], dtype=float) + shift

    def finish(ax, title, ylabel=None):
        ax.set_yscale("log")
        ax.grid(True, which="both", ls=":", alpha=0.35)
        ax.set_xticks(range(len(ticks)))
        ax.set_xticklabels([f"{o:g}" for o in ticks], fontsize=9)
        ax.set_xlabel("output_reg", fontsize=10)
        ax.set_title(title, fontsize=10.5)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=8.5)

    ax = axes[-1][0]
    for variant, cmap_name, shift in (("ude", "Purples", -0.07), ("node", "Blues", 0.07)):
        df = data[variant]
        if df is None:
            continue
        fin = finals(df[(df.shooting_mode == "single") & (~df.cl)])
        if fin is None:
            continue
        colour = plt.get_cmap(cmap_name)(0.72)
        ax.scatter(positions(fin.output_reg, shift), fin.mse_train.values, s=26, color=colour,
                   alpha=0.5, edgecolors="none", zorder=3)
        median = fin.groupby("output_reg").mse_train.median()
        ax.plot(positions(median.index, shift), median.values, "o-", color=colour, lw=2.2, ms=7,
                zorder=4, label=f"{variant.upper()} SS")
    finish(ax, "SS  (no windows / no penalty)", "final training MSE")

    if ncol > 1:
        ax = axes[-1][1]
        for variant, cmap_name, base in (("ude", "Purples", -0.04), ("node", "Blues", 0.04)):
            df = data[variant]
            if df is None:
                continue
            for k, stages in enumerate(sorted(df.num_curriculum_steps.unique())):
                if stages <= 0:
                    continue
                fin = finals(df[(df.shooting_mode == "single") & (df.cl)
                                & (df.num_curriculum_steps == stages)])
                if fin is None:
                    continue
                colour = plt.get_cmap(cmap_name)(0.45 + 0.4 * k)
                shift = base + (-0.10 if k == 0 else 0.10)
                ax.scatter(positions(fin.output_reg, shift), fin.mse_train.values, s=26,
                           color=colour, alpha=0.5, edgecolors="none", zorder=3)
                median = fin.groupby("output_reg").mse_train.median()
                ax.plot(positions(median.index, shift), median.values, "o-" if k == 0 else "s--",
                        color=colour, lw=2.2, ms=7, zorder=4,
                        label=f"{variant.upper()} CL {int(stages)} stages")
        finish(ax, "CL  (stages instead of windows)")
    for j in range(2, ncol):
        axes[-1][j].axis("off")

    fig.suptitle(f"Hyperparameter interactions, {args.grid}x{args.grid} grid ({experiment})\n"
                 "parallel lines => no interaction  |  crossing lines => the best penalty depends "
                 "on window count  |  dots = seeds, shared y axis", fontsize=13, y=0.995)
    save(fig, outdir, f"rd_hpscan_interaction_{args.grid}x{args.grid}", args.dpi, not args.no_pdf)


if __name__ == "__main__":
    main()
