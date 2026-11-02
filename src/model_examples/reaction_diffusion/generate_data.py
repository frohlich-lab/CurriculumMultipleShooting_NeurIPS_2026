"""Generate ground-truth reaction-diffusion (FitzHugh-Nagumo) trajectories, one per grid size.

Solves the *true* PDE with a fine adaptive solver (matching APHYNITY's RK45), samples at the
APHYNITY effective step dt=0.1 over [0, T_VAL], and saves a flattened trajectory to
``src/data_dir/reaction_diffusion_<N>/ground_truth.npz`` (keys: ``t``, ``data``). The UDE and NODE
variants at a given N share the same npz; the loader slices train (t <= 2.5) / val (t > 2.5).

Run on a COMPUTE NODE (not the login node), e.g. ``srun ... python3 <this file>``.
"""
from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import config

from src.common import data_dir
from src.model_examples.reaction_diffusion.setup_reaction_diffusion_model import (
    DOMAIN_L,
    DT,
    IC_SEED,
    NOISE_STD_FRAC,
    REACTION_DIFFUSION_SIZES,
    T_INIT,
    T_VAL,
    N_TRAIN_POINTS,
    N_FORECAST_POINTS,
    make_random_field,
    true_rhs,
)


def generate_for_size(N: int) -> None:
    dx = DOMAIN_L / N
    # APHYNITY: raw uniform field at t=T_INIT (-0.5); integrate through the 0.5-unit burn-in and only
    # SAVE from t=0 onward. data[0] = X0 = state at t=0 (smoothed), matching get_X0().
    # Exactly DT-spaced so the training split lands on N_TRAIN_POINTS by construction.
    n_steps = N_TRAIN_POINTS + N_FORECAST_POINTS
    ts = DT * jnp.arange(n_steps)                 # t = 0 .. T_VAL, step DT
    y0 = make_random_field(N)                     # the field at t=T_INIT

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(true_rhs),
        diffrax.Tsit5(),
        t0=float(T_INIT),                         # start integrating at t=-0.5
        t1=float(T_VAL),
        dt0=1e-3,
        y0=y0,
        args={"grid_size": N, "dx": dx},
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
        max_steps=1_000_000,
    )
    data = np.asarray(sol.ys)                     # (n_steps, 2 * N**2), data[0] = X0 at t=0

    if NOISE_STD_FRAC > 0:
        # per-channel (u, v) noise, scaled by each channel's std over the trajectory
        half = N * N
        rng = np.random.default_rng(IC_SEED)
        for lo, hi in ((0, half), (half, 2 * half)):
            std = data[:, lo:hi].std()
            data[:, lo:hi] += rng.normal(0.0, NOISE_STD_FRAC * std, size=data[:, lo:hi].shape)

    out_dir = Path(data_dir) / f"reaction_diffusion_{N}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ground_truth.npz"
    np.savez(out_path, t=np.asarray(ts), data=data)
    print(f"Saved: {out_path}  (shape {data.shape})")


def main() -> None:
    config.update("jax_enable_x64", True)
    for N in REACTION_DIFFUSION_SIZES:
        generate_for_size(N)


if __name__ == "__main__":
    main()
