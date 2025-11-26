DYNAMICAL_MODEL = {
    "lotka_volterra",
    # "sir"
}

# MLP
MLP_WIDTHS = {
    "range": (
        # 5,  # UDE paper -> activation changed to RBF, max_lrate to 1e-1
        10,
        # 20,
    ),
    "central_value": 10,
}

MLP_DEPTHS = {
    "range": (
        # 1,
        2,
        # 5
    ),
    "central_value": 2,
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
        5e-2,
        1e-2,
        5e-1,
        1e-1,
    ),
    "central_value": 1e-2,
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
    "exp",
    # "lin",
    # "cos",  # default
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
        1e-1,
        1e0,
        1e1,
        1e2,
        3e2,  # add more intermediate values after rescaling regularisation for time-normalisation runs
        5e2,
        1e3,
        1e4,
    ),
    "central_value": 0,  # use 1e0 for UDEs
}


# Shooting parameters (shooting mode is decided by TRAINING_SCHEMES)
SHOOTING_WINDOW_LENGTHS =  {
    "range": (
        4,  # 21 windows
        5,  # 16 windows
        6,  # 13 windows
        8,  # 9 windows
        14  # 5 windows
    ),
    "central_value": 8,
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
    "central_value": 10,
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
    "central_value": 9,
}

MAX_CURRICULUM_EPOCHS = {
    2000,
}

RESTART_OPT_CURRICULUM = {
    # True,  # restart optimiser at each curriculum stage
    False,  # continue training with the same optimiser
}

# Dynamic Parameters estimation settings
DYNAMIC_PARAMS_SCALES = {
    "linear",  # NODE
    # "log10",  # UDE
}

MULTISTART_DEVIATIONS = {
    0,
    # 1,
    # 2
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
    True,
    # False,
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


# Training settings (UDE, MLP, UDE+MLP)
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

N_STARTS = 100  # number of multi-starts

N_EPOCHS_ADAM = 6000  # default from UDE paper: 200 / SIR: 40000-100000
N_EPOCHS_BFGS = 10000

EXPERIMENT_NAME = "multiple_NODE_ls_26112025"
