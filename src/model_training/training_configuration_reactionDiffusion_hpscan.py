# Reaction-diffusion (FitzHugh-Nagumo) benchmark, based on APHYNITY (arXiv:2010.04456).
# Hyperparameter scan on the 8x8 grid (state dim 128).
# Two variants: hybrid UDE (known diffusion + ConvNet reaction) and vanilla augmented NODE.
# Full factorial over output_reg x shooting_window_length x multiple_shooting_penalty x
# num_curriculum_steps (FACTORIAL_SCAN_PARAMS); all other options are single committed values.
# Axes: 2 model names x 4 training schemes (single_bfgs disabled) x factorial grid x N_STARTS seeds.
#
# To activate: copy this file over src/model_training/training_configuration.py.

DYNAMICAL_MODEL = {
    "reaction_diffusion_ude_8",
    "reaction_diffusion_node_8",
}

# NN architecture -- ConvNet (APHYNITY): mlp_layer_width -> hidden channels, mlp_depth -> conv layers
MLP_WIDTHS = {
    "range": (16,),      # ConvNet hidden channels (APHYNITY)
    "central_value": 16,
}

MLP_DEPTHS = {
    "range": (3,),       # number of conv layers (APHYNITY 3-layer ConvNet)
    "central_value": 3,
}

INIT_SCALES = {
    0.1,
}

USE_BIASES = {
    True,
}

USE_FINAL_BIASES = {
    True,
}

MLP_ACTIVATIONS = {
    "relu",              # APHYNITY uses ReLU
}

MLP_FINAL_ACTIVATIONS = {
    "identity",
}

USE_DROPOUT = {
    False,
}

INPUT_DROPOUT_RATES = {
    0.8,
}

HIDDEN_DROPOUT_RATES = {
    0.5,
}

TRAINING_SCHEMES = {
    "single",
    "single_CL",
    "multiple",
    "multiple_CL",
    # "single_bfgs",  # removed: second-order (fides/BFGS) over a 2*N**2-dim problem is unstable/expensive here
}

# Optimiser
OPTIMISERS = {
    "adam",
}

MAX_LEARNING_RATES = {
    # Tuned on the 8x8 LR sweep (5 seeds, {1e-4,3e-4,1e-3,3e-3,1e-2}), selected on TRAIN MSE:
    # 3e-3 is best or statistically tied for 7 of 8 method x variant cells.
    # (Only NODE-MS prefers 1e-2, but it diverges at every LR, so it is not a real optimum.)
    "range": (3e-3,),
    "central_value": 3e-3,
}

LEARNING_RATE_SPANS = {
    1e0,
}

LEARNING_RATE_DECAYS = {
    0.9 ** 0,            # 1.0 -> no decay
}

WARMUP_FCTS = {
    0,
}

OPT_STEPS = {
    0,
}

OPT_MULT = {
    0,
}

OPT_SCHED_TYPES = {
    "lin",
}

# Regularisation
MLP_REG_EXPS = {
    1.0,
}

MLP_REG_STRENGTHS = {
    "range": (0,),
    "central_value": 0,
}

OUTPUT_REG = {
    # HP-SCAN AXIS. UDE only; NODE forced to 0 in prune_config.
    "range": (1.0, 10.0, 100.0),
    "central_value": 10.0,
}

# Shooting parameters (shooting mode is decided by TRAINING_SCHEMES)
SHOOTING_WINDOW_LENGTHS = {
    # HP-SCAN AXIS. 26 train points: wl=7 -> 5 windows, wl=5 -> 7 windows.
    "range": (5, 7),
    "central_value": 7,
}

SHOOTING_IV_INITS = {
    "z",                 # zeros (+ eps); first window = u0 guess
}

MULTIPLE_SHOOTING_PENALTIES = {
    # HP-SCAN AXIS. Applies to MS and (since the prune override was lifted) CMS.
    "range": (0.1, 1.0, 10.0),
    "central_value": 1.0,
}

# Curriculum learning (decided by TRAINING_SCHEMES)
NUM_CURRICULUM_STEPS = {
    # HP-SCAN AXIS - the "stages" counterpart of window count, for single_CL (which has no windows).
    "range": (5, 7),
    "central_value": 5,
}

MAX_CURRICULUM_EPOCHS = {
    "range": (1667,),    # ~first third of the 5000-epoch ADAM budget
    "central_value": 1667,
}

RESTART_OPT_CURRICULUM = {
    False,
}

# Dynamic parameters estimation settings
# UDE: diffusion coefficients (a, b) estimated on log10 scale (numerically stable for the small
# O(1e-3) coefficients; repo convention for UDE mechanistic params). NODE: forced linear in prune_config.
DYNAMIC_PARAMS_SCALES = {
    "log10",
}

# UDE: perturb the diffusion (a, b) init so parameter recovery is a real test rather than "stayed at
# truth". deviation=2.0 => log10(true) +/- 2, i.e. +/-2 orders of magnitude (0.01x-100x of truth),
# matching the CMS paper (App C.3). NODE: forced to 0 in prune_config (no mechanistic params).
MULTISTART_DEVIATIONS = {
    2.0,
}

DYNAMIC_PARAMS_PERTURB_METHODS = {
    "lh",                # latin hypercube
}

# Node augmentation (vanilla augmented NODE): extra latent channels; forced to 0 for the UDE variant
NODE_AUGMENTATIONS = {
    "range": (2,),
    "central_value": 2,
}

# Hessian options -- MUST stay False (dense P x P Hessian over a 2*N**2 state is infeasible)
COMPUTE_HESSIAN_METRICS = {
    False,
}

# Second optimiser (classical) -- only used by single_bfgs
SECOND_OPTS = {
    "fides",
}

FREEZE_MECHANISTIC_PARAMS_ADAM = {
    False,
}
FREEZE_MECHANISTIC_PARAMS_OPTX = {
    False,
}
FREEZE_NN_ADAM = {
    False,
}
FREEZE_NN_OPTX = {
    False,
}

# Training settings (UDE, NODE, MLP) -- forced per model name in prune_config; dedup collapses the rest
TRAINING_SETTINGS = {
    "UDE",
    "NODE",
}

MLPEXP_TARGETS = {
    "None",
}

# Normalisation -- both off. Time horizon is O(1) here, so time normalisation is unnecessary; keeping
# it off means the estimated diffusion coefficients (a, b) are in physical units (raw == de-normalised),
# directly comparable to the truth (1e-3, 5e-3) for parameter-recovery evaluation.
NORMALISE_DATA = {
    False,
}

NORMALISE_TIMEPOINTS = {
    False,
}

SAVE_LANDSCAPE_MODELS = {
    False,
}

NOISE_LEVELS = {
    None,
}

TIME_SPAN = {
    None,
}

SCALE_INITIAL_VALUES = {
    None,
}

N_STARTS = 10  # number of multi-starts (seeds)

N_EPOCHS_ADAM = 5000
N_EPOCHS_BFGS = 1000  # unused (single_bfgs removed from TRAINING_SCHEMES)

EXPERIMENT_NAME = "reaction_diffusion_hpscan_25072026"

# Enumerate these axes as a FULL FACTORIAL (all combinations) rather than the default
# one-at-a-time linear scan -- window length and lambda_cont are expected to interact.
FACTORIAL_SCAN_PARAMS = frozenset({
    "output_reg",
    "shooting_window_length",
    "multiple_shooting_penalty",
    "num_curriculum_steps",
})
