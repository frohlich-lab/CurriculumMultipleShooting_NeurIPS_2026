DYNAMICAL_MODEL = {
    "pendulum",
}

# MLP
MLP_WIDTHS = {
    "range": (
        50,  # Homotopy paper
    ),
    "central_value": 50,
}

MLP_DEPTHS = {
    "range": (
        2,
    ),
    "central_value": 2,  # Homotopy paper
}

INIT_SCALES = {
    0.1,
}

USE_BIASES = {
    True,
    # False,
}

USE_FINAL_BIASES = {
    True,
    # False,
}


MLP_ACTIVATIONS = {
    # "tanh",
    # "relu",
    "swish",
    # "gelu",
    # "rbf"
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
    # "bfgs",
    # "single",
    # "single_bfgs",
    # "single_CL",
    "multiple",
    # "multiple_CL",
}

# Optimiser
OPTIMISERS = {
    "adam",
    # "adamw",
}

MAX_LEARNING_RATES = {
    "range": (
        1e-3,
        5e-3,
        1e-2,
        5e-2,
        1e-1,
        # 5e-1,
    ),
    "central_value": 5e-2,  # 1e-2 for everything apart from MS, use 5e-2 for MS
}

LEARNING_RATE_SPANS = {
    1e0,  # initial learning rate = max learning rate
}

LEARNING_RATE_DECAYS = {
    0.9**0,  # 1.0  -> no decay
}

WARMUP_FCTS = {
    0,  # -> no warmup; in combination with lrate_span=1 and decay=1 this results in a constant lr schedule
}

OPT_STEPS = {
    0,  # -> defaults to a single schedule
}

OPT_MULT = {
    0,  # -> defaults to a single schedule
}

OPT_SCHED_TYPES = {
    "lin", # linear schedule (default)
    # "cos",  # cosine
    # "exp",  # exponential
}


# Regularisation
MLP_REG_EXPS = {
    1.0,  # L1
    # 2.0,
}

MLP_REG_STRENGTHS = {
    "range": (
        0,
        # 1e-1,
        # 1e0,
        # 1e1,
        # 1e2
    ),
    "central_value": 0,
}

# OUTPUT_REG = {
#     0,
#     # changed with 1e-6 premultiplier given issues with snakemake small wildcards
#     1e0, # corresponds to 1e-6
#     1e1,  # corresponds to 1e-5
#     1e2,  # corresponds to 1e-4
#     1e3,  # corresponds to 1e-3
#     # 1e-2,
#     # 1e-1,
#     # 1e0,
#     # 1e1,
#     # 1e2,
#     # 1e3,
#     # 1e4,
#     # 1e5,
#     # 1e6,
# }

OUTPUT_REG = {
    "range": (
        0,
        # 1e-4,
        # 1e-3,
        # 1e-2,
        1e-1,
        1e0,
        1e1,
        1e2,
        # # 3e2,  # add more intermediate values after rescaling regularisation for time-normalisation runs
        5e2,
        1e3,
        # 3e3,
        5e3,
        1e4,
        # 3e4,
        5e4,
    ),
    "central_value": 0,  # use 1e0 for UDEs, 0 for NODEs // starting with 0 for SIR
    # LV: 1e3 for single_BFGS, single_CL & multiple; 5e3 for multiple_CL
}


# Shooting parameters (shooting mode is decided by TRAINING_SCHEMES)
SHOOTING_WINDOW_LENGTHS =  {
    "range": (
        4,  # 21 windows
        5,  # 16 windows
        6,  # 13 windows
        8,  # 9 windows
        14,  # 5 windows
        21,  # third data length
        30  # half data length approximately
    ),
    "central_value": 5,  # LV: optimal 5 for both multiple and multiple_CL
}

SHOOTING_IV_INITS = {
    "z",  # zeros
    # "g",  # guess
    # "ru",  # random uniform
}

MULTIPLE_SHOOTING_PENALTIES =  {
    "range": (
        0,
        1e-3,  # note: 2e-3 used to work decently
        1e-2,
        1e-1,
        1e0,
        1e1,
        1e2,
        1e3
    ),
    "central_value": 10,  # use 10 for everything apart from multiple_CL (tried 0; now use 1e-2)
    # LV: optimal 10 for multiple; 1.0 for multiple_CL
}

# Batch size -- unused, only for window-batch, DISABLED
# BATCH_SIZES = {
#     64,
# }

# Curriculum learning (decided by TRAINING_SCHEMES)
NUM_CURRICULUM_STEPS =  {
    "range": (
        3,
        5,   # matches Homotopy paper and corresponding multiple+CL
        9,
        13,
        16,
        21
    ),
    "central_value": 13,  # LV: optimal 13 for single_CL
}

MAX_CURRICULUM_EPOCHS = {
    "range": (
        2000,
    ),
    "central_value": 2000,
}

RESTART_OPT_CURRICULUM = {
    # True,  # restart optimiser at each curriculum stage
    False,  # continue training with the same optimiser
}

# Dynamic Parameters estimation settings
# NODEs: set parameters to 0 and use linear scale
DYNAMIC_PARAMS_SCALES = {
    "linear",  # NODE
}

MULTISTART_DEVIATIONS = {
    0, # run only 0 for NODEs
}

DYNAMIC_PARAMS_PERTURB_METHODS = {
    "lh",  # latin_hypercube
    # "su",  # sample_uniform
    # "agn",  # add_gaussian_noise
}

# Node augmentation
NODE_AUGMENTATIONS = {
    "range": (
        0,
        # 1,
        # 2,
        # 3
    ),
    "central_value": 0,
}

# Hessian Options
COMPUTE_HESSIAN_METRICS = {
    # True,
    False,
}

# Second optimiser (classical)
SECOND_OPTS = {
    "fides",
    # "optx",
}

# Option to freeze the mechanistic parameters and only train the neural network component
FREEZE_MECHANISTIC_PARAMS_ADAM = {
    # True,
    False
}
FREEZE_MECHANISTIC_PARAMS_OPTX= {
    # True,
    False
}

FREEZE_NN_ADAM= {
    # True,
    False
}

FREEZE_NN_OPTX= {
    # True,
    False
}


# Training settings (UDE, NODE, MLP)
TRAINING_SETTINGS = {
    # "UDE",
    "NODE",
    # "MLP",
}

MLPEXP_TARGETS = {
    "RHS",
    # "interaction"
}

# Normalisation -- time: yes, data: no
NORMALISE_DATA = {
    # True,
    False,
}

NORMALISE_TIMEPOINTS = {
    True,
    # False,
}

SAVE_LANDSCAPE_MODELS = {
    # True,
    False,
}

N_STARTS = 100  # number of multi-starts

N_EPOCHS_ADAM = 6000
N_EPOCHS_BFGS = 10000

EXPERIMENT_NAME = "multiple_pendulum_NODE_ls_09122025"
