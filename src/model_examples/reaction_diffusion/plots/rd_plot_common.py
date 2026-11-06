"""Shared helpers for the reaction-diffusion figures.

Every path is derived from src.common (results_dir / plots_dir) and every constant from
setup_reaction_diffusion_model, so no script hardcodes a results location, an experiment
name, a grid list or a parameter value.
"""
import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from src.common import plots_dir, results_dir
from src.model_examples.reaction_diffusion.setup_reaction_diffusion_model import (
    DIFF_A, DIFF_B, REACTION_DIFFUSION_SIZES,
)

VARIANTS = ("ude", "node")

# (label, shooting_mode, curriculum_learning, colormap) -- one colour family per method, reused
# across every figure so a method keeps its identity.
METHODS = [
    ("SS", "single", False, "Purples"),
    ("CL", "single", True, "Blues"),
    ("MS", "multiple", False, "Greens"),
    ("CMS", "multiple", True, "Oranges"),
]
# grid resolution -> shade within the method's colormap
SHADE = {4: 0.32, 8: 0.52, 16: 0.74, 32: 0.96}


def state_dim(grid: int) -> int:
    """Observed state dimension of the 2-channel field on a grid x grid mesh."""
    return 2 * grid * grid


def default_experiment() -> str:
    """The experiment name from the active training configuration."""
    from src.model_training.training_configuration import EXPERIMENT_NAME
    return EXPERIMENT_NAME


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--experiment", default=None,
                       help="experiment name under results/ (default: EXPERIMENT_NAME from the "
                            "active training configuration)")
    parser.add_argument("--grids", type=int, nargs="+", default=list(REACTION_DIFFUSION_SIZES),
                       help="grid sizes to include")
    parser.add_argument("--outdir", default=None, type=Path,
                       help="directory for the figures (default: plots/<experiment>)")
    parser.add_argument("--dpi", type=int, default=300, help="raster resolution")
    parser.add_argument("--no-pdf", action="store_true", help="skip the vector copy")
    return parser


def resolve(args: argparse.Namespace):
    """-> (experiment name, output directory). Creates the directory."""
    experiment = args.experiment or default_experiment()
    outdir = Path(args.outdir) if args.outdir else plots_dir / experiment
    outdir.mkdir(parents=True, exist_ok=True)
    return experiment, outdir


def train_dir(experiment: str, variant: str, grid: int) -> Path:
    return results_dir / experiment / f"reaction_diffusion_{variant}_{grid}" / "train"


def load_runs(experiment: str, variant: str, grid: int,
              usecols: Optional[Iterable[str]] = None) -> Optional[pd.DataFrame]:
    """Concatenate every per-run CSV for one (variant, grid). None if nothing has landed yet.

    Adds a boolean `cl` column, since curriculum_learning round-trips through CSV as a string.
    Partial results are expected while a sweep is still running, so unreadable files are skipped.
    """
    wanted = set(usecols) if usecols else None
    frames: List[pd.DataFrame] = []
    for path in sorted(train_dir(experiment, variant, grid).glob("*.csv")):
        try:
            frames.append(pd.read_csv(path, usecols=(lambda c: c in wanted) if wanted else None))
        except Exception:
            continue
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["cl"] = df["curriculum_learning"].astype(str).isin(["True", "true", "1"])
    return df.sort_values("epoch") if "epoch" in df else df


def final_per_seed(df: pd.DataFrame, method: Optional[tuple] = None) -> pd.DataFrame:
    """Last logged epoch of every seed, optionally filtered to one (shooting_mode, cl) method."""
    if method is not None:
        shooting_mode, cl = method
        df = df[(df.shooting_mode == shooting_mode) & (df.cl == cl)]
    if df.empty:
        return df
    return df.sort_values("epoch").groupby("multistart_job").tail(1)


def param_error(a, b) -> np.ndarray:
    """Mechanistic-parameter recovery: mean |log10(estimate / truth)| over the diffusion
    coefficients, i.e. orders of magnitude off (0 = exact). Defined for the UDE variant only --
    the NODE variant has no mechanistic parameters."""
    a = np.abs(np.asarray(a, dtype=float)).clip(1e-12)
    b = np.abs(np.asarray(b, dtype=float)).clip(1e-12)
    return 0.5 * (np.abs(np.log10(a / DIFF_A)) + np.abs(np.log10(b / DIFF_B)))


def save(fig, outdir: Path, stem: str, dpi: int = 300, pdf: bool = True) -> None:
    png = outdir / f"{stem}.png"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    if pdf:
        fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    print(f"SAVED {png}" + (f" (+ {stem}.pdf)" if pdf else ""))
