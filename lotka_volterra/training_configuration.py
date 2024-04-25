DYNAMICAL_MODEL = {"lotka_volterra"}

MLP_WIDTHS = {1, 5, 10, 50}

MLP_DEPTHS = {1, 2, 3, 5, 10}

MLP_ACTIVATIONS = {
    "tanh",
    "relu",
    "swish"
}

OPTIMISERS = {
    "adam",
    "adamw",
}

USE_EARLY_STOP = {
    True,
    # False,
}

N_EPOCHS = 1000

