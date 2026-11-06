"""Factorial hyperparameter scan: output_reg x lambda_cont heatmaps per method and window length.

Rows = variant x shooting method, columns = shooting window length, cells = output_reg (y) against
the continuity penalty (x), coloured by median final training MSE. Every panel shares one log
colour scale so cells are comparable across the whole figure; the scale is clipped, because a
diverged cell can otherwise stretch it over enough decades to flatten everything else. Clipped
cells are marked and their true value is still printed, so nothing is hidden.

Only the multiple-shooting methods span all three axes: single shooting has no windows and no
continuity penalty, and the NODE variant has output_reg forced to zero.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

from src.model_examples.reaction_diffusion.plots.rd_plot_common import (
    add_common_args, load_runs, resolve, save,
)

USECOLS = ["epoch", "mse_train", "shooting_mode", "curriculum_learning", "multistart_job",
           "output_reg", "shooting_window_length", "multiple_shooting_penalty"]
# (variant, label, shooting_mode, curriculum_learning, colormap)
ROWS = [("ude", "UDE  MS", "multiple", False, "Greens"),
        ("ude", "UDE  CMS", "multiple", True, "Oranges"),
        ("node", "NODE  MS", "multiple", False, "Greens"),
        ("node", "NODE  CMS", "multiple", True, "Oranges")]


def cell_value(df, shooting_mode, cl, window, output_reg, penalty):
    """(median final training MSE, seed count) for one factorial cell; nan if not run yet."""
    sub = df[(df.shooting_mode == shooting_mode) & (df.cl == cl)
             & (df.shooting_window_length == window)
             & np.isclose(df.output_reg, output_reg)
             & np.isclose(df.multiple_shooting_penalty, penalty)]
    if sub.empty:
        return np.nan, 0
    fin = sub.sort_values("epoch").groupby("multistart_job").tail(1)
    return float(fin.mse_train.median()), int(fin.multistart_job.nunique())


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--grid", type=int, default=8, help="grid size the scan was run on")
    parser.add_argument("--clip", type=float, default=1.0,
                       help="upper limit of the shared colour scale; cells above it saturate and "
                            "are marked (0 or negative disables clipping)")
    args = parser.parse_args()
    experiment, outdir = resolve(args)

    data = {v: load_runs(experiment, v, args.grid, USECOLS) for v in ("ude", "node")}
    # axis levels are discovered from the data, so the figure follows whatever was scanned
    present = [df for df in data.values() if df is not None]
    if not present:
        raise SystemExit(f"no results found for {experiment} at {args.grid}x{args.grid}")
    windows = sorted({int(w) for df in present for w in df.shooting_window_length.unique() if w > 0})
    penalties = sorted({float(p) for df in present
                        for p in df.multiple_shooting_penalty.unique() if p > 0})
    oregs = {v: sorted({float(o) for o in df.output_reg.unique()}) if df is not None else []
             for v, df in data.items()}

    # pass 1: build every grid so a single colour scale can cover all panels
    grids, counts = {}, {}
    for i, (variant, _, shooting_mode, cl, _) in enumerate(ROWS):
        df = data[variant]
        levels = oregs[variant]
        per_window, per_window_n = [], []
        for window in windows:
            block = np.full((len(levels), len(penalties)), np.nan)
            block_n = np.zeros_like(block, dtype=int)
            if df is not None:
                for a, output_reg in enumerate(levels):
                    for b, penalty in enumerate(penalties):
                        block[a, b], block_n[a, b] = cell_value(
                            df, shooting_mode, cl, window, output_reg, penalty)
            per_window.append(block)
            per_window_n.append(block_n)
        grids[i], counts[i] = per_window, per_window_n

    finite = np.concatenate([b[np.isfinite(b)] for blocks in grids.values() for b in blocks]
                            + [np.array([])])
    if not finite.size:
        raise SystemExit("results found but no complete factorial cell yet")
    clip = args.clip if args.clip and args.clip > 0 else finite.max()
    vmin, vmax = finite.min(), min(clip, max(finite.max(), finite.min() * 10))
    norm = LogNorm(vmin=vmin, vmax=vmax)
    print(f"shared colour scale: {vmin:.2e} .. {vmax:.2e}"
          f"   true max: {finite.max():.2e}   cells above clip: {int((finite > vmax).sum())}")

    fig, axes = plt.subplots(len(ROWS), len(windows), figsize=(5.5 * len(windows), 14),
                            squeeze=False, gridspec_kw={"hspace": 0.45, "wspace": 0.18})
    for i, (variant, row_label, _, _, cmap_name) in enumerate(ROWS):
        levels = oregs[variant]
        row_values = np.concatenate([b[np.isfinite(b)] for b in grids[i]] + [np.array([])])
        best = row_values.min() if row_values.size else None
        for j, window in enumerate(windows):
            ax = axes[i][j]
            block, block_n = grids[i][j], counts[i][j]
            im = ax.imshow(block, cmap=cmap_name, norm=norm, aspect="auto", origin="lower")
            ax.set_xticks(range(len(penalties)))
            ax.set_xticklabels([f"{p:g}" for p in penalties])
            ax.set_yticks(range(len(levels)))
            ax.set_yticklabels([f"{o:g}" for o in levels])
            ax.set_xlabel(r"$\lambda_{cont}$", fontsize=10)
            if j == 0:
                ax.set_ylabel("output_reg", fontsize=10)
            ax.set_title(f"{row_label}   window length {window}", fontsize=11)
            for a in range(block.shape[0]):
                for b in range(block.shape[1]):
                    value = block[a, b]
                    if not np.isfinite(value):
                        ax.text(b, a, "--", ha="center", va="center", fontsize=8, color="0.5")
                        continue
                    shade = (np.log10(min(value, vmax) / vmin)
                             / max(np.log10(vmax / vmin), 1e-9))
                    ax.text(b, a, ("^ " if value > vmax else "")
                            + f"{value:.1e}\nn={block_n[a, b]}", ha="center", va="center",
                            fontsize=7.5, color="white" if shade > 0.55 else "0.15")
                    if best is not None and np.isclose(value, best):
                        ax.add_patch(Rectangle((b - .5, a - .5), 1, 1, fill=False,
                                               edgecolor="black", lw=2.5, zorder=5))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label(
                "median train MSE", fontsize=8)

    fig.suptitle(f"Factorial hyperparameter scan, {args.grid}x{args.grid} grid ({experiment})\n"
                 "median final TRAINING MSE  |  black box = best cell in that row  |  "
                 f"one shared colour scale clipped at {vmax:g} (^ = above clip)",
                 fontsize=13, y=0.995)
    save(fig, outdir, f"rd_hpscan_heatmap_{args.grid}x{args.grid}", args.dpi, not args.no_pdf)

    for i, (variant, row_label, shooting_mode, cl, _) in enumerate(ROWS):
        df = data[variant]
        if df is None:
            continue
        best, where = np.inf, None
        for window in windows:
            for output_reg in oregs[variant]:
                for penalty in penalties:
                    value, n = cell_value(df, shooting_mode, cl, window, output_reg, penalty)
                    if np.isfinite(value) and value < best:
                        best, where = value, (window, output_reg, penalty, n)
        if where:
            print(f"  {row_label}: best={best:.3e} at window={where[0]} output_reg={where[1]:g} "
                  f"lambda_cont={where[2]:g} (n={where[3]})")


if __name__ == "__main__":
    main()
