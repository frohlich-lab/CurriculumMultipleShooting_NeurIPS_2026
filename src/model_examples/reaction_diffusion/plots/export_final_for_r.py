"""Export the final 100-seed reaction-diffusion run to one tidy CSV for the manuscript R figures.

Output: plots/<experiment>/data/rd_final_traces.csv, one row per (model, tag, seed, epoch), with
the columns the manuscript's R code (sciml_ode_training/rsrc, parse_best_new) expects:
loss_train / loss_val are the logged mse_train / mse_val, start_guess is the seed, and a / b are
the diffusion coefficients (UDE only; NaN for NODE). Method labels follow the manuscript:
Adam = single shooting, CL = single + curriculum, MS = multiple shooting, CMS = multiple +
curriculum. Run inside the container (needs pandas):
    python3 -m src.model_examples.reaction_diffusion.plots.export_final_for_r [--experiment ...]
"""
import argparse

import pandas as pd

from src.common import plots_dir
from src.model_examples.reaction_diffusion.plots.rd_plot_common import load_runs

COLS = ["epoch", "mse_train", "mse_val", "a", "b", "shooting_mode", "curriculum_learning",
        "multistart_job", "max_curriculum_epochs"]
TAG = {("single", False): "Adam", ("single", True): "CL",
       ("multiple", False): "MS", ("multiple", True): "CMS"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", default="reaction_diffusion_final_19102026")
    p.add_argument("--grid", type=int, default=64)
    args = p.parse_args()
    frames = []
    for variant in ("ude", "node"):
        df = load_runs(args.experiment, variant, args.grid, COLS)
        if df is None:
            raise SystemExit(f"no results for {variant}")
        df["model"] = variant.upper()
        df["tag"] = [TAG[(s, c)] for s, c in zip(df.shooting_mode, df.cl)]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True).rename(columns={
        "mse_train": "loss_train", "mse_val": "loss_val", "multistart_job": "start_guess",
        "max_curriculum_epochs": "n_cl_epochs"})
    out = out[["model", "tag", "start_guess", "epoch", "loss_train", "loss_val", "a", "b",
               "n_cl_epochs"]]
    n = out.groupby(["model", "tag"]).start_guess.nunique()
    assert (n == 100).all(), n
    dest = plots_dir / args.experiment / "data"
    dest.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest / "rd_final_traces.csv", index=False)
    print(f"wrote {len(out)} rows -> {dest / 'rd_final_traces.csv'}")
    print(out.groupby(["model", "tag"]).agg(seeds=("start_guess", "nunique"),
                                            epochs=("epoch", "nunique")).to_string())


if __name__ == "__main__":
    main()
