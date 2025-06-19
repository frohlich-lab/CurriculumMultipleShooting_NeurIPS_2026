DYNAMICAL_MODEL = {
    "lotka_volterra",
    # "sir"
}

# MLP
MLP_WIDTHS = {
    # 1,
    5,  # UDE paper -> activation changed to RBF, max_lrate to 1e-1
    # 10,
    # 50
}

MLP_DEPTHS = {
    # 1,
    2,
    # 3,
    # 5,
    # 10
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
    # "swish",
    # "gelu",
    "rbf"
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
    "single",
    "single_bfgs",
    "single_CL",
    "multiple",
    "multiple_CL",
}

# Optimiser
OPTIMISERS = {
    "adam",
    # "adamw",
}

MAX_LEARNING_RATES = {
    1e-1,
    # 1e-2,
    # 1e-3,
}

LEARNING_RATE_SPANS = {
    1e0,  # initial learning rate = max learning rate
    # 1e1,
    # 1e2,
    # 1e3,
}

LEARNING_RATE_DECAYS = {
    0.9**0,  # 1.0  -> no decay
    # 0.999,
    # 0.9**1,
    # 0.9**2,
    # 0.9**3,
}

WARMUP_FCTS = {
    # 0.4,
    # 0.2,
    # 0.1,
    # 0.025,  # 100 warmup epochs for 4000 training epochs
    # 0.05,
    # 1e-2,
    # 1e-3,
    0,  # -> no warmup; in combination with lrate_span=1 and decay=1 this results in a constant lr schedule
}

OPT_STEPS = {
    0,  # -> defaults to a single schedule
    # 1,
    # 2,
    # 10,
}

OPT_MULT = {
    0,  # -> defaults to a single schedule
    # 1,
    # 2,
    # 3,
}

OPT_SCHED_TYPES = {
    "exp",
    # "lin",
    # "cos",  # default
}


# Regularisation
MLP_REG_EXPS = {
    1.0,
    # 2.0,
}

MLP_REG_STRENGTHS = {
    0,
    # 1e1,
    # 1e2,
    # 1e3,
    # 1e4,
    # 1e5,
    # 1e6,
}

OUTPUT_REG = {
    0,
    # changed with 1e-6 premultiplier given issues with snakemake small wildcards
    # 1e0, # corresponds to 1e-6
    # 1e1,  # corresponds to 1e-5
    # 1e2,  # corresponds to 1e-4
    # 1e3,  # corresponds to 1e-3
    # 1e-2,
    # 1e-1,
    # 1e0,
    # 1e1,
    # 1e2,
    # 1e3,
    # 1e4,
    # 1e5,
    # 1e6,
}

# Shooting parameters (shooting mode is decided by TRAINING_SCHEMES)
SHOOTING_WINDOW_LENGTHS = {
    6,  # 13 windows
    # # 7,
    8,  # 9 windows
    # 10,
    14,  # 5 windows
}

SHOOTING_IV_INITS = {
    "z",  # zeros
    # "g",  # guess
    # "ru",  # random uniform
}

MULTIPLE_SHOOTING_PENALTIES = {
    0.0,
    # 1e-1,
    # 1e-2,
    2e-3,
}

# Batch size -- unused, only for window-batch, DISABLED
# BATCH_SIZES = {
#     64,
# }

# Curriculum learning (decided by TRAINING_SCHEMES)
NUM_CURRICULUM_STEPS = {
    # 5,   # matches Homotopy paper and corresponding multiple+CL
    9
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
    # "linear",
    "log10",
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
    0,
    # 1,
}

# Hessian Options
COMPUTE_HESSIAN_METRICS = {
    True,
    # False,
}

# Second optimiser (classical)
SECOND_OPTS = {
    # "fides",
    "optx",
}

# Option to freeze the mechanistic parameters and only train the neural network component
FREEZE_MECHANISTIC_PARAMS_ADAM = {
    True,
    # False
}
FREEZE_MECHANISTIC_PARAMS_OPTX= {
    True,
    # False
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
    "UDE",
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

N_STARTS = 50  # number of multi-starts

N_EPOCHS_ADAM = 4000  # default from UDE paper: 2000 / SIR: 40000-100000
N_EPOCHS_BFGS = 10000

EXPERIMENT_NAME = "nemo_udefrozen"
