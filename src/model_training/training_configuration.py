DYNAMICAL_MODEL = {
    "lotka_volterra",
    # "sir"
}

# MLP
MLP_WIDTHS = {
    # 1,
    # 5,
    10,
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
    "swish"
}


MLP_FINAL_ACTIVATIONS = {
   "identity",
}


USE_SKIP_CONNECTIONS = {
    False,
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

# Optimiser
OPTIMISERS = {
    "adam",
    # "adamw",
}

MAX_LEARNING_RATES = {
    # 1e-1,
    1e-2,
    # 1e-3,
}

LEARNING_RATE_SPANS = {
    # 1e0,
    1e1,
    # 1e2,
    # 1e3,
}

LEARNING_RATE_DECAYS = {
    0.9**0,
    # 0.9**1,
    # 0.9**2,
    # 0.9**3,
}

WARMUP_FCTS = {
    # 0.4,
    # 0.2,
    0.1,
    # 0.05,
    # 1e-2,
    # 1e-3,
}

OPT_STEPS = {
    # 1,
    # 2,
    10,
}

OPT_MULT = {
    # 1,
    2,
    # 3,
}

# # Other optimisation hyperparams
# LookAhead disabled
# USE_LOOKAHEAD = {
#     "True",
#     "False",
# }
#
# LOOKAHEAD_SYNC = 5

USE_EARLY_STOP = {
    # True,
    False,
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
    1e-4,
    1e-3,
    1e-2,
    # 1e-1,
    # 1e0,
    # 1e1,
    # 1e2,
    # 1e3,
    # 1e4,
    # 1e5,
    # 1e6,
}


# Shooting modes
SHOOTING_MODES = {
    "single",
    "multiple",
}

SHOOTING_WINDOW_LENGTHS = {
    7,
    # 10,
}

SHOOTING_IV_INITS = {
    "z",  # zeros
    # "g",  # guess
    # "ru",  # random uniform
}

MULTIPLE_SHOOTING_PENALTIES = {
    0.0,
    1e-1,
    1e-2,
    1e-3,
}

# Batch size -- unused, only for window-batch, DISABLED
# BATCH_SIZES = {
#     64,
# }

# Curriculum learning
CURRICULUM_LEARNING = {
    True,
    False,
}

NUM_CURRICULUM_STEPS = {
    10
}

MAX_CURRICULUM_EPOCHS = {
    2000,
}

SERIALISE_CURRICULUM_MODELS = {
    False,
}

# Dynamic Parameters estimation settings
DYNAMIC_PARAMS_SCALES = {
    # "linear",
    "log10",
}

MULTISTART_DEVIATIONS = {
    0,
    1,
    2
}

DYNAMIC_PARAMS_PERTURB_METHODS = {
    "su",  # sample_uniform
    # "agn",  # add_gaussian_noise
}

# Node augmentation
NODE_AUGMENTATIONS = {
    0,
    # 1,
}

# Hessian Options
HESSIAN_PRECONDITION = {
    # True,
    False,
}

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
    "UDE",
    # "MLP",
}

MLPEXP_TARGETS = {
    "RHS",
    # "interaction"
}

# Normalisation
NORMALISE_DATA = {
    True,
    # False,
}

NORMALISE_TIMEPOINTS = {
    True,
    # False,
}

N_STARTS = 50  # number of multi-starts

MECHANISTIC_FIT_EPOCHS = 0
N_EPOCHS = 4000  # default from UDE paper: 2000 / SIR: 40000-100000
MAX_ITER_OPTX_FINETUNE = 10000

EXPERIMENT_NAME = "nemo_v1"
