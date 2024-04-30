DYNAMICAL_MODEL = {"lotka_volterra"}

MLP_WIDTHS = {1, 5, 10, 50}

MLP_DEPTHS = {1, 2, 3, 5, 10}

MLP_ACTIVATIONS = {
    "tanh",
    "relu",
    "swish"
}

USE_EARLY_STOP = {
    True,
    # False,
}

OPTIMISERS = {
    "adam",
    "adamw",
}

MAX_LEARNING_RATES = {
    1e-2,
    1e-3,
}

LEARNING_RATE_SPANS = {
    # 1e0,
    1e1,
    1e2,
    # 1e3,
}

LEARNING_RATE_DECAYS = {
    0.9**0,
    0.9**1,
    # 0.9**2,
    # 0.9**3,
}

WARMUP_FCTS = {
    # 0.4,
    0.2,
    0.1,
    # 0.05,
    # 1e-2,
    # 1e-3,
}

N_EPOCHS = 1000

REG_EXPS = {
    1.0,
    2.0,
}

REG_STRENGTHS = {
    1e1,
    1e2,
    1e3,
    1e4,
    1e5,
    1e6,
}

