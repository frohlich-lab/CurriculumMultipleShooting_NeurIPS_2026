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
  vector field every evaluation using batch/spatial statistics, which can destabilise the solve
  (APHYNITY integrates with a discrete fixed-step RK solver, where this is benign).
"""
import os
from typing import Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import diffrax
from jaxtyping import Array

from src.model_examples.base_ude_function import BaseUDEFunction
from src.model_training.base_losses import get_masked_mse, get_mse
from src.model_training.model_util import process_mechanistic_params


# --- Physical constants (from the APHYNITY paper) ---------------------------------------------
# Oscillatory (spiral-wave) FitzHugh-Nagumo regime:
#     du/dt = Du Lap(u) + u - u^3/3 - v
#     dv/dt = Dv Lap(v) + eps * (u + a - b v)
# The fixed point sits on the middle branch of the cubic, so the medium is oscillatory and a
# gradient-like initial condition induces a phase defect around which a spiral rotates.
DOMAIN_L = 100.0    # spiral needs room: a small domain synchronises to a uniform oscillation
DIFF_A = 1.0        # Du  (mechanistic parameter of the UDE variant)
DIFF_B = 1.0        # Dv  (mechanistic parameter of the UDE variant)
FHN_A = 0.1         # reaction offset a
FHN_B = 1.0         # inhibitor decay b
RD_EPSILON = 0.005  # time-scale separation on the recovery equation
RD_NORMALISE_CHANNELS = os.environ.get("RD_NORMALISE_CHANNELS", "false").lower() in ("1","true","yes")
                    # Weight each channel's contribution to the loss by 1/std(channel).
                    # The activator and inhibitor live on very different scales (std 1.63 vs 0.32
                    # on the spiral dataset), so an unweighted MSE is ~96% determined by u and
                    # barely constrains v -- even though v is the inhibitor that drives the
                    # oscillation. Weighting equalises them. The MODEL stays in physical units;
                    # only the objective changes, so (a, b) remain directly interpretable.
RD_OBSERVED = os.environ.get("RD_OBSERVED", "uv")
                    # which channels enter the loss: "uv" (both) or "u" (activator only, so the
                    # inhibitor is a latent state that has to be inferred from the dynamics).
                    # Only the LOSS and the stored data are affected -- the model always integrates
                    # both channels, and the t=0 initial condition is taken as known.
RD_BC = "neumann"   # boundary condition for BOTH the mechanistic Laplacian and the ConvNet padding.
                    # "periodic" is the original APHYNITY setup; "neumann" (zero-flux) is required
                    # for the spiral-wave regime, where a periodic domain would wrap the wave.
# Horizon is aligned with the shooting windows: 2 spiral rotations across 5 windows of length 8
# (0.4 rotations per window, ~18 samples per rotation). Reduced from 3 rotations: the 3-rotation
# task could not be learned (no configuration beat a constant predictor) and a shorter horizon is
# both an easier extrapolation and ~1/3 cheaper to integrate. The spiral rotation period measured at
# N=64 is ~258, markedly faster than the 0-D limit cycle (~426) since the core sets the rotation.
SPIRAL_PERIOD = 258.2
N_PERIODS_TRAIN = 2
T_INIT = 0.0        # the gradient initial condition IS the observed state; no burn-in
T_TRAIN = N_PERIODS_TRAIN * SPIRAL_PERIOD           # 2 rotations
N_TRAIN_POINTS = 36                                 # 5 windows x length 8, overlapping by one point
DT = T_TRAIN / (N_TRAIN_POINTS - 1)                 # exact spacing: t[35] == T_TRAIN
# Forecast one further rotation, rounded to a whole number of DT steps so the time grid stays
# exactly DT-spaced (building it with linspace instead gives 35 train points, not 36).
N_FORECAST_POINTS = int(round(SPIRAL_PERIOD / DT))
# Defined off the same grid as the save points: T_TRAIN + N_FORECAST*DT is algebraically
# identical but differs in the last bit, which puts ts[-1] outside t1 and trips diffrax.
T_VAL = (N_TRAIN_POINTS + N_FORECAST_POINTS - 1) * DT
NOISE_STD_FRAC = 0.0  # APHYNITY data are clean; set > 0 to add Gaussian noise (fraction of per-channel std)
IC_SEED = 0         # fixed seed -> data identical across training seeds

# Grid sizes registered as separate dynamical models (state-space-size scan)
REACTION_DIFFUSION_SIZES = (64,)   # below ~64 the spiral is under-resolved at L=100


def periodic_laplacian(f: Array, dx: float) -> Array:
    """5-point finite-difference Laplacian with periodic boundary conditions (circular padding)."""
    return (
        jnp.roll(f, 1, axis=0) + jnp.roll(f, -1, axis=0)
        + jnp.roll(f, 1, axis=1) + jnp.roll(f, -1, axis=1)
        - 4.0 * f
    ) / dx ** 2


def neumann_laplacian(f: Array, dx: float) -> Array:
    """5-point Laplacian with Neumann (zero-flux) boundaries via replicated ghost cells."""
    fp = jnp.pad(f, 1, mode="edge")
    return (
        fp[2:, 1:-1] + fp[:-2, 1:-1] + fp[1:-1, 2:] + fp[1:-1, :-2] - 4.0 * f
    ) / dx ** 2


def laplacian(f: Array, dx: float, bc: str = None) -> Array:
    """Laplacian under the configured boundary condition (see ``RD_BC``).

    The ConvNet padding in ``src/ude_problem/conv_net.py`` must use the matching mode, otherwise
    the learned reaction term assumes different boundaries from the mechanistic diffusion it is
    added to.
    """
    bc = RD_BC if bc is None else bc
    if bc == "periodic":
        return periodic_laplacian(f, dx)
    if bc == "neumann":
        return neumann_laplacian(f, dx)
    raise ValueError(f"unknown boundary condition: {bc!r}")


CONV_PADDING = {"periodic": "wrap", "neumann": "edge"}

# --- Explicit-solver step-size cap for the diffusion term --------------------------------------
# The 5-point Laplacian on an N x N grid has real eigenvalues in [-8/dx**2, 0], so the stiffest
# mode of `D * Lap` sits at `-D * 8/dx**2`.  Tsit5's stability region reaches to about -3.3 on the
# negative real axis, giving `h < 3.3 / (D * 8/dx**2)`.
#
# Why this has to be enforced rather than left to the PI controller: `shooting_iv_init='z'` starts
# the shooting windows from a SPATIALLY UNIFORM field, whose Laplacian is exactly zero.  The stiff
# modes are then unexcited, contribute nothing to the embedded error estimate, and the controller
# happily takes steps ~9x past the stability limit.  The forward solve survives that (those modes
# have no amplitude to grow), but reverse mode propagates the FULL Jacobian, so every high-k mode
# is multiplied by |R(h*lambda)| >> 1 at each step and the gradient overflows.  Measured on the
# 64x64 spiral, seed 0 (a=8.64, b=5.93, window span 103.3, stability limit h<0.1166):
#
#     start          steps   mean h   h/h_stab   max |dy_end/dy0|   non-finite
#     1e-6 uniform      95   1.0872      9.32x   -                  8192 / 8192
#     observed field   895   0.1154      0.99x   3.48                        0
#     1e-6 + dtmax cap 991   0.1042      0.89x   2.29                        0
#
# i.e. the cap alone (same initial values, same everything else) removes the blow-up.
TSIT5_REAL_STABILITY_RADIUS = 3.3
# Headroom below the bare stability limit.  The bound above accounts for the diffusion operator
# only; the ConvNet reaction adds its own (complex) eigenvalues on top, and the estimate assumes
# the asymptotic 5-point spectrum.  0.9 was already sufficient in the measurement above; 0.8 keeps
# a 25% margin on top of that.
#
# Cost scales as 1/factor, and it is paid on EVERY window, including ones the controller was
# already integrating stably -- window 0 above sits at 0.99x the bare limit, so it is the safety
# factor, not the fix, that sets its step count. Measured at seed 0 (a=8.64), steps per window:
#     uncapped   895 (window 0)  /   95 (uniform windows, UNSTABLE in reverse mode)
#     factor 0.5      1787       / 1778
#     factor 0.8      ~1117      / ~1111
# i.e. 0.5 would have doubled the cost of an already-correct solve for no extra safety.
DTMAX_SAFETY_FACTOR = 0.8


# Fixed point (middle branch of the cubic) and the extent of the 0-D limit cycle, which set the
# gradient ranges below. u* solves  -u^3/3 + (1 - 1/b) u - a/b = 0; take the real root.
_ROOTS = np.roots([-1 / 3.0, 0.0, 1.0 - 1.0 / FHN_B, -FHN_A / FHN_B])
_REAL = _ROOTS[np.abs(_ROOTS.imag) < 1e-9].real
U_STAR = float(_REAL[np.argmin(np.abs(_REAL))])
V_STAR = (U_STAR + FHN_A) / FHN_B
U_LO, U_HI = -1.998, 1.992          # measured limit-cycle extent at eps = 0.005
V_LO, V_HI = -0.685, 0.696


def make_initial_field(N: int) -> jnp.ndarray:
    """Gradient-like initial condition, flattened as ``[u.ravel(), v.ravel()]``.

    ``u`` varies horizontally and ``v`` vertically, each spanning the limit cycle and centred on
    the fixed point, so every grid point starts at a different phase of the cycle and the single
    point where ``(u, v) == (u*, v*)`` is a topological phase defect at the centre of the domain.
    The spiral winds around that defect. Deterministic: no RNG involved.
    """
    r_u = max(U_HI - U_STAR, U_STAR - U_LO)
    r_v = max(V_HI - V_STAR, V_STAR - V_LO)
    ramp = np.linspace(-1.0, 1.0, N)
    u0 = U_STAR + r_u * ramp[None, :] * np.ones((N, 1))     # varies along x (columns)
    v0 = V_STAR + r_v * ramp[:, None] * np.ones((1, N))     # varies along y (rows)
    return jnp.asarray(np.concatenate([u0.ravel(), v0.ravel()]))


# Backwards-compatible alias: the IC is no longer random, but callers import this name.
make_random_field = make_initial_field


def get_X0(grid_size: int) -> jnp.ndarray:
    """Observed initial condition X0 = state at t=0 (post 0.5-unit burn-in), read from the npz.

    Guarantees the training IC equals the data's t=0 state exactly (both come from the same file).
    """
    return jnp.asarray(np.load(_ground_truth_path(grid_size))["data"][0])


def _reaction(u_ch: Array, v_ch: Array, epsilon: float = RD_EPSILON) -> Tuple[Array, Array]:
    """True FitzHugh-Nagumo reaction terms (used only for ground-truth data generation).

    ``epsilon`` is the canonical FHN time-scale separation, placed on the recovery (inhibitor)
    equation in the usual way:

        du/dt = a Lap(u) + u - u^3 - k - v
        dv/dt = b Lap(v) + eps * (u - v)

    eps = 1 recovers the APHYNITY parameterisation exactly, so existing data is unchanged.
    Decreasing eps slows the inhibitor relative to the activator, which is what makes the system
    excitable rather than simply relaxing.
    """
    r_u = u_ch - u_ch ** 3 / 3.0 - v_ch
    r_v = epsilon * (u_ch + FHN_A - FHN_B * v_ch)
    return r_u, r_v


def true_rhs(t, y, args):
    """Right-hand side of the *true* reaction-diffusion PDE (for data generation).

    ``args`` may carry an ``epsilon`` entry; it defaults to RD_EPSILON (= 1) so that callers which
    do not set it reproduce the original APHYNITY system bit-for-bit.
    """
    N, dx = args["grid_size"], args["dx"]
    u_ch = y[:N * N].reshape(N, N)
    v_ch = y[N * N:].reshape(N, N)
    r_u, r_v = _reaction(u_ch, v_ch, args.get("epsilon", RD_EPSILON))
    du = DIFF_A * laplacian(u_ch, dx) + r_u
    dv = DIFF_B * laplacian(v_ch, dx) + r_v
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

    def max_stable_dt(self) -> Array:
        """Tsit5 stability bound for the known diffusion term, at the CURRENT (a, b).

        `a` and `b` are learned, so this has to be recomputed from the live parameters rather than
        fixed once: a run that drifts to a larger diffusion coefficient needs a smaller step.
        Gradient-blocked -- the step size is a numerical detail, not part of the model.
        """
        a, b = process_mechanistic_params(self.dynamic_parameters, self.options)
        dx = self.options["extra_args"]["dx"]
        stiffest = jnp.maximum(jnp.max(jnp.array([a, b])), 1e-30) * (8.0 / dx ** 2)
        return jax.lax.stop_gradient(
            DTMAX_SAFETY_FACTOR * TSIT5_REAL_STABILITY_RADIUS / stiffest
        )

    def __call__(self, t, u, args):
        a, b = process_mechanistic_params(self.dynamic_parameters, self.options)
        N = self.options["extra_args"]["grid_size"]
        dx = self.options["extra_args"]["dx"]

        field = u[:self.state_dimension].reshape(2, N, N)          # channels (u, v)
        reaction = self.mlp(field)                                 # ConvNet: (2, N, N) -> (2, N, N)

        du = a * laplacian(field[0], dx) + reaction[0]
        dv = b * laplacian(field[1], dx) + reaction[1]
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


_CHANNEL_WEIGHT_CACHE = {}


def channel_weights(state_dimension: int):
    """Per-component loss weights (1/std of the owning channel), or ones if disabled.

    Computed from the TRAINING portion of the ground truth, so it is a property of the data and
    not of any model. Cached as a NUMPY array: caching a jnp array would capture a tracer if the
    first call happens inside a jit/cond trace, and that tracer then leaks into later traces
    (jax.errors.UnexpectedTracerError). NumPy constants promote cleanly wherever they are used.
    """
    n_obs = n_observed(state_dimension)
    key = (state_dimension, n_obs, RD_NORMALISE_CHANNELS)
    if key not in _CHANNEL_WEIGHT_CACHE:
        if not RD_NORMALISE_CHANNELS:
            w = np.ones(n_obs)
        else:
            grid = int(round((state_dimension / 2) ** 0.5))
            npz = np.load(_ground_truth_path(grid))
            data, t = np.asarray(npz["data"]), np.asarray(npz["t"])
            train = data[t <= T_TRAIN + 1e-9]
            half = train.shape[1] // 2
            s_u = float(train[:, :half].std())
            s_v = float(train[:, half:].std())
            if n_obs <= half:                       # u only
                w = np.full(n_obs, 1.0 / s_u)
            else:
                w = np.concatenate([np.full(half, 1.0 / s_u),
                                    np.full(n_obs - half, 1.0 / s_v)])
        _CHANNEL_WEIGHT_CACHE[key] = w
    return _CHANNEL_WEIGHT_CACHE[key]


def n_observed(state_dimension: int) -> int:
    """Number of leading state components that are observed.

    The state is laid out as [u.ravel(), v.ravel()], so the activator is the leading half and
    "u"-only observation is a contiguous slice.
    """
    return state_dimension if RD_OBSERVED == "uv" else state_dimension // 2


def reaction_diffusion_base_loss(solution: jnp.ndarray, data, state_dimension: int):
    w = channel_weights(state_dimension)
    pred = solution[:, :n_observed(state_dimension)]
    return get_mse(pred=pred * w, data=data * w)


def reaction_diffusion_base_loss_masked(solution: jnp.ndarray, data, state_dimension: int, mask: jnp.ndarray):
    w = channel_weights(state_dimension)
    pred = solution[:, :n_observed(state_dimension)]
    return get_masked_mse(pred=pred * w, data=data * w, mask=mask)


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
    # keep only the observed channels (see RD_OBSERVED); the model still integrates both
    data = data[:, :n_observed(data.shape[1])]

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
