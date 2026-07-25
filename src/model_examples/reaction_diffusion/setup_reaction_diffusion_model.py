"""Reaction-diffusion (FitzHugh-Nagumo) benchmark, based on APHYNITY (arXiv:2010.04456).

Two channels ``(u, v)`` evolve on an ``N x N`` periodic grid:

    du/dt = a * Laplacian(u) + (u - u**3 - k - v)
    dv/dt = b * Laplacian(v) + (u - v)

The state is stored flattened as ``[u.ravel(), v.ravel()]`` (length ``2 * N**2``). Scanning the
grid resolution ``N`` therefore scans the state-space size (``2 * N**2``). Two model variants are
provided (mirroring the Lotka-Volterra UDE/NODE pair):

* ``ReactionDiffusionUDEFunction`` - hybrid UDE: known diffusion + a ConvNet learning the reaction.
* ``ReactionDiffusionNODEFunction`` - vanilla augmented NODE: a ConvNet models the full RHS, with
  ``node_augmentation`` extra latent channels (Dupont et al. augmentation, as channels).

Simulation/data parameters follow the APHYNITY paper (domain [-1, 1]^2, a=1e-3, b=5e-3, k=5e-3,
effective dt=0.1, train horizon 2.5 = 25 steps, forecast/validation over the next 25 steps).

Divergences from APHYNITY (arXiv:2010.04456; github.com/yuan-yin/APHYNITY)
-------------------------------------------------------------------------
This benchmark reuses APHYNITY's *system* to stress-test CMS training strategies at high state
dimension; it is NOT a reproduction of their forecasting numbers. Deliberate differences:

* Data regime: a SINGLE trajectory (one fixed IC) with a temporal interpolation/extrapolation
  split, vs their 1600 train / 320 test trajectories over many ICs (multi-IC generalisation).
* Training: full-batch fit with the CMS strategies (single/CL/MS/CMS) for 5000 epochs, vs their
  minibatch Adam for 50000 epochs with the APHYNITY min-norm objective (min ||F_a||). In place of
  that min-norm constraint we use output regularisation on the NN reaction output (UDE variant) to
  curb it from absorbing the known diffusion; NODE keeps output_reg=0.
* Metric: our variance/uncertainty is over 10 multistart seeds (median-reported); their reported
  +/- std is over the 320-trajectory test set from a single training run (no seed loop).
* Diffusion coefficients (a, b): estimated on log10 scale via multistart (numerically stable for
  the small O(1e-3) coefficients), vs their bounded-linear ``sigmoid(theta) * 1e-2``.
* ConvNet (see ``src/ude_problem/conv_net.py``): kept intentionally as a biased, circular-padded,
  BatchNorm-free 3-layer 3x3 ConvNet (hidden=16). APHYNITY's ``ConvNetEstimator`` instead uses
  ``BatchNorm2d(track_running_stats=False)`` after conv layers 1-2, ``bias=False`` on those two
  convs, and ZERO padding on the NN convs (circular padding only in the physics Laplacian). We do
  NOT replicate BatchNorm: inside an adaptive diffrax ODE right-hand side it would renormalise the
  vector field every evaluation using batch/spatial statistics and can destabilise the solve
  (APHYNITY integrates with a discrete fixed-step RK solver, where this is benign).
"""
import os
from typing import Tuple, Union

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import diffrax
from jaxtyping import Array

from src.model_examples.base_ude_function import BaseUDEFunction
from src.model_training.base_losses import get_masked_mse, get_mse
from src.model_training.model_util import process_mechanistic_params


# --- Physical constants (from the APHYNITY paper) ---------------------------------------------
DOMAIN_L = 2.0      # physical domain is Omega = [-1, 1]^2, side length 2; dx = DOMAIN_L / N
DIFF_A = 1e-3       # diffusion coefficient of u
DIFF_B = 5e-3       # diffusion coefficient of v
RD_K = 5e-3         # reaction offset k
DT = 0.1            # APHYNITY effective time step (RK45 at 1e-3, subsampled by 100)
T_INIT = -0.5       # APHYNITY: random field is set at t=-0.5 and integrated; X0 = state at t=0
                    # (0.5-time-unit burn-in that smooths the raw uniform field before the observed run)
T_TRAIN = 2.5       # training horizon (25 steps at dt=0.1)
T_VAL = 5.0         # full horizon; forecast/validation covers the next 25 steps
NOISE_STD_FRAC = 0.0  # APHYNITY data are clean; set > 0 to add Gaussian noise (fraction of per-channel std)
IC_SEED = 0         # fixed seed -> data identical across training seeds

# Grid sizes registered as separate dynamical models (state-space-size scan)
REACTION_DIFFUSION_SIZES = (4, 8, 16, 32)


def periodic_laplacian(f: Array, dx: float) -> Array:
    """5-point finite-difference Laplacian with periodic boundary conditions (circular padding)."""
    return (
        jnp.roll(f, 1, axis=0) + jnp.roll(f, -1, axis=0)
        + jnp.roll(f, 1, axis=1) + jnp.roll(f, -1, axis=1)
        - 4.0 * f
    ) / dx ** 2


def make_random_field(N: int) -> jnp.ndarray:
    """The raw uniform random field set at t=T_INIT (-0.5), flattened as ``[u.ravel(), v.ravel()]``.

    APHYNITY samples ``X_init in [0, 1]^(2 x N x N)`` uniformly at t=-0.5, then integrates; the
    OBSERVED initial condition X0 is the state at t=0 (see ``generate_data`` and ``get_X0``). This is
    only the integration start, NOT X0. Uses NumPy's RNG (deterministic, x64-independent).
    """
    rng = np.random.default_rng(IC_SEED)
    return jnp.asarray(rng.uniform(size=2 * N * N))


def get_X0(grid_size: int) -> jnp.ndarray:
    """Observed initial condition X0 = state at t=0 (post 0.5-unit burn-in), read from the npz.

    Guarantees the training IC equals the data's t=0 state exactly (both come from the same file).
    """
    return jnp.asarray(np.load(_ground_truth_path(grid_size))["data"][0])


def _reaction(u_ch: Array, v_ch: Array) -> Tuple[Array, Array]:
    """True FitzHugh-Nagumo reaction terms (used only for ground-truth data generation)."""
    r_u = u_ch - u_ch ** 3 - RD_K - v_ch
    r_v = u_ch - v_ch
    return r_u, r_v


def true_rhs(t, y, args):
    """Right-hand side of the *true* reaction-diffusion PDE (for data generation)."""
    N, dx = args["grid_size"], args["dx"]
    u_ch = y[:N * N].reshape(N, N)
    v_ch = y[N * N:].reshape(N, N)
    r_u, r_v = _reaction(u_ch, v_ch)
    du = DIFF_A * periodic_laplacian(u_ch, dx) + r_u
    dv = DIFF_B * periodic_laplacian(v_ch, dx) + r_v
    return jnp.concatenate([du.ravel(), dv.ravel()])


class ReactionDiffusionUDEFunction(BaseUDEFunction):
    """Hybrid UDE: known diffusion (a*Lap(u), b*Lap(v)) + ConvNet reaction term.

    The mechanistic parameters are the diffusion coefficients ``(a, b)``; the ConvNet
    (``self.mlp``) maps the 2-channel field to a 2-channel reaction contribution.
    """

    dynamic_parameters: Array
    mlp: Union[eqx.nn.MLP, eqx.Module]
    options: dict
    state_dimension: int
    state_delimiter: int
    regularise_network: bool

    def __call__(self, t, u, args):
        a, b = process_mechanistic_params(self.dynamic_parameters, self.options)
        N = self.options["extra_args"]["grid_size"]
        dx = self.options["extra_args"]["dx"]

        field = u[:self.state_dimension].reshape(2, N, N)          # channels (u, v)
        reaction = self.mlp(field)                                 # ConvNet: (2, N, N) -> (2, N, N)

        du = a * periodic_laplacian(field[0], dx) + reaction[0]
        dv = b * periodic_laplacian(field[1], dx) + reaction[1]
        derivatives = jnp.concatenate([du.ravel(), dv.ravel()])

        if self.regularise_network:
            d_nn_reg = jnp.array([jnp.linalg.norm(reaction)])
            derivatives = jnp.concatenate([derivatives, d_nn_reg])
        return derivatives


class ReactionDiffusionNODEFunction(BaseUDEFunction):
    """Vanilla augmented NODE: a ConvNet models the full RHS over all channels.

    ``node_augmentation`` extra latent channels are appended (Dupont et al. augmentation). The
    ConvNet consumes and produces ``C = 2 + node_augmentation`` channels; only the first two
    (observed) channels are compared to data.
    """

    dynamic_parameters: Array
    mlp: Union[eqx.nn.MLP, eqx.Module]
    options: dict
    state_dimension: int
    state_delimiter: int
    regularise_network: bool

    def __call__(self, t, u, args):
        N = self.options["extra_args"]["grid_size"]
        # state_delimiter = (2 + node_augmentation) * N * N (observed + augmented channels)
        n_channels = self.state_delimiter // (N * N)
        field = u[:self.state_delimiter].reshape(n_channels, N, N)
        nn_out = self.mlp(field)                                   # (C, N, N) -> (C, N, N)
        derivatives = nn_out.ravel()

        if self.regularise_network:
            # regularise the AUGMENTED channels' dynamics (dz/dt), not the observed (u, v) RHS - to
            # keep the augmentation minimal. Empty (norm 0) when node_augmentation == 0.
            d_nn_reg = jnp.array([jnp.linalg.norm(derivatives[self.state_dimension:self.state_delimiter])])
            derivatives = jnp.concatenate([derivatives, d_nn_reg])
        return derivatives


def reaction_diffusion_base_loss(solution: jnp.ndarray, data, state_dimension: int):
    return get_mse(pred=solution[:, :state_dimension], data=data)


def reaction_diffusion_base_loss_masked(solution: jnp.ndarray, data, state_dimension: int, mask: jnp.ndarray):
    return get_masked_mse(pred=solution[:, :state_dimension], data=data, mask=mask)


# UTILS - DATA LOADING
def _ground_truth_path(grid_size: int) -> str:
    return os.path.join(
        "src", "data_dir", f"reaction_diffusion_{grid_size}", "ground_truth.npz"
    )


def get_training_validation_data(grid_size: int):
    """Load the cached ground-truth trajectory and apply the APHYNITY temporal split.

    Train = time points with ``t <= T_TRAIN`` (25 steps); validation = the next 25 forecast steps
    ``T_TRAIN < t <= T_VAL``. Returns the standard 6-tuple used by the other examples.
    """
    npz = np.load(_ground_truth_path(grid_size))
    t = np.asarray(npz["t"])
    data = np.asarray(npz["data"])

    train_mask = t <= T_TRAIN + 1e-9
    t_train, train_data = t[train_mask], data[train_mask]
    t_val, val_data = t[~train_mask], data[~train_mask]

    t_save_train = diffrax.SaveAt(ts=jnp.asarray(t_train))
    t_save_val = diffrax.SaveAt(ts=jnp.asarray(t_val))
    t_span_train = [0.0, float(T_TRAIN)]
    t_span_val = [0.0, float(T_VAL)]

    return (
        jnp.asarray(train_data), t_save_train, t_span_train,
        jnp.asarray(val_data), t_save_val, t_span_val,
    )
