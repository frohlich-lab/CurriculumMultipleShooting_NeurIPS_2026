import itertools as itt
from typing import Dict, List

from src.common import scan_attributes
try:  # opt-in, per-config: axes to enumerate as a full factorial instead of one-at-a-time
    from src.model_training.training_configuration import FACTORIAL_SCAN_PARAMS
except ImportError:
    FACTORIAL_SCAN_PARAMS = frozenset()
try:  # opt-in: apply the tuned per-method reaction-diffusion hyperparameters
    from src.model_training.training_configuration import RD_USE_METHOD_CONFIG
except ImportError:
    RD_USE_METHOD_CONFIG = False
from src.model_training.training_configuration import (
    COMPUTE_HESSIAN_METRICS,
    DYNAMIC_PARAMS_PERTURB_METHODS,
    DYNAMIC_PARAMS_SCALES,
    DYNAMICAL_MODEL,
    INIT_SCALES,
    LEARNING_RATE_DECAYS,
    LEARNING_RATE_SPANS,
    MAX_CURRICULUM_EPOCHS,
    MAX_LEARNING_RATES,
    MLP_ACTIVATIONS,
    MLP_DEPTHS,
    # USE_LOOKAHEAD, LOOKAHEAD_SYNC,
    MLP_REG_EXPS,
    MLP_REG_STRENGTHS,
    MLP_WIDTHS,
    MLPEXP_TARGETS,
    MULTIPLE_SHOOTING_PENALTIES,
    MULTISTART_DEVIATIONS,
    N_EPOCHS_ADAM,
    N_EPOCHS_BFGS,
    N_STARTS,
    NODE_AUGMENTATIONS,
    NOISE_LEVELS,
    NORMALISE_DATA,
    NORMALISE_TIMEPOINTS,
    NUM_CURRICULUM_STEPS,
    OPT_MULT,
    OPT_SCHED_TYPES,
    OPT_STEPS,
    OPTIMISERS,
    OUTPUT_REG,
    RESTART_OPT_CURRICULUM,
    SAVE_LANDSCAPE_MODELS,
    SCALE_INITIAL_VALUES,
    SECOND_OPTS,
    SHOOTING_IV_INITS,
    SHOOTING_WINDOW_LENGTHS,
    TIME_SPAN,
    # MLP_FINAL_ACTIVATIONS,
    # USE_DROPOUT, INPUT_DROPOUT_RATES, HIDDEN_DROPOUT_RATES,
    TRAINING_SCHEMES,
    TRAINING_SETTINGS,
    # FREEZE_MECHANISTIC_PARAMS_ADAM, FREEZE_MECHANISTIC_PARAMS_OPTX, FREEZE_NN_ADAM, FREEZE_NN_OPTX,
    USE_BIASES,
    USE_FINAL_BIASES,
    WARMUP_FCTS,
)

linear_hyperparameters = {
    # NN architecture
    "mlp_layer_width": MLP_WIDTHS,
    "mlp_depth": MLP_DEPTHS,
    # Optimisation
    "max_lrate": MAX_LEARNING_RATES,
    # Regularisation
    "mlp_reg_strength": MLP_REG_STRENGTHS,
    "output_reg": OUTPUT_REG,
    # Shooting modes
    "shooting_window_length": SHOOTING_WINDOW_LENGTHS,
    "multiple_shooting_penalty": MULTIPLE_SHOOTING_PENALTIES,
    # Curriculum Learning
    "num_curriculum_steps": NUM_CURRICULUM_STEPS,
    "max_curriculum_epochs": MAX_CURRICULUM_EPOCHS,
    # NODE augmentation
    "node_augmentation": NODE_AUGMENTATIONS,
}

product_hyperparameters = {
    # MLP
    "init_scale": INIT_SCALES,
    "use_bias": USE_BIASES,
    "use_final_bias": USE_FINAL_BIASES,
    "activation_fn_name": MLP_ACTIVATIONS,
    # "final_activation_fn_name": MLP_FINAL_ACTIVATIONS,
    # "use_dropout": USE_DROPOUT,
    # "input_dropout_rate": INPUT_DROPOUT_RATES,
    # "hidden_dropout_rate": HIDDEN_DROPOUT_RATES,
    # Training scheme, NN optimiser + scheduling
    "training_scheme": TRAINING_SCHEMES,
    "optimiser": OPTIMISERS,
    "lrate_span": LEARNING_RATE_SPANS,
    "lrate_decay": LEARNING_RATE_DECAYS,
    "warmup_fct": WARMUP_FCTS,
    "opt_steps": OPT_STEPS,
    "opt_mult": OPT_MULT,
    "opt_sched_type": OPT_SCHED_TYPES,
    # Regularisation
    "mlp_reg_exp": MLP_REG_EXPS,
    # Shooting modes
    "shooting_iv_init": SHOOTING_IV_INITS,
    # Curriculum learning
    "restart_opt_curriculum": RESTART_OPT_CURRICULUM,
    # Dynamic param settings
    "dynamic_params_scale": DYNAMIC_PARAMS_SCALES,
    "multistart_deviation": MULTISTART_DEVIATIONS,
    "dynamic_params_perturb_method": DYNAMIC_PARAMS_PERTURB_METHODS,
    # Hessian
    "compute_hessian_metrics": COMPUTE_HESSIAN_METRICS,
    # Second (classical) optimiser
    "second_opt": SECOND_OPTS,
    # Freeze params
    # "freeze_mechanistic_params_adam": FREEZE_MECHANISTIC_PARAMS_ADAM,
    # "freeze_mechanistic_params_optx": FREEZE_MECHANISTIC_PARAMS_OPTX,
    # "freeze_nn_adam": FREEZE_NN_ADAM,
    # "freeze_nn_optx": FREEZE_NN_OPTX,
    # Normalisation
    "normalise_data": NORMALISE_DATA,
    "normalise_timepoints": NORMALISE_TIMEPOINTS,
    # Training setting
    "training_setting": TRAINING_SETTINGS,
    "mlpexp_target": MLPEXP_TARGETS,
    "save_landscape_models": SAVE_LANDSCAPE_MODELS,
    "noise_level": NOISE_LEVELS,
    "time_span": TIME_SPAN,
    "scale_initial_values": SCALE_INITIAL_VALUES,
}

# Per-method reaction-diffusion hyperparameters, selected on the 8x8 factorial scan (5 seeds).
#   output_reg  : highest value whose fit degradation vs the best is NOT significant
#                 (paired Wilcoxon on final train MSE); 100 degrades significantly everywhere.
#   window len / lambda_cont : best cell on train MSE, chosen JOINTLY with output_reg -- they
#                 interact (UDE-MS flips from wl=7,lambda=0.1 to wl=5,lambda=1 once oreg=10).
#   stages      : num_curriculum_steps, for single+CL only (CL has no shooting windows).
# Keys are (variant, shooting_mode, curriculum_learning). NODE output_reg stays forced to 0.
# Applied only when the active config sets RD_USE_METHOD_CONFIG.
RD_METHOD_CONFIG = {
    ("ude",  "single",   False): {"output_reg": 10.0},
    ("ude",  "single",   True):  {"output_reg": 10.0, "num_curriculum_steps": 5},
    ("ude",  "multiple", False): {"output_reg": 10.0, "shooting_window_length": 5,
                                  "multiple_shooting_penalty": 1.0},
    ("ude",  "multiple", True):  {"output_reg": 10.0, "shooting_window_length": 7,
                                  "multiple_shooting_penalty": 1.0},
    ("node", "single",   True):  {"num_curriculum_steps": 5},
    ("node", "multiple", False): {"shooting_window_length": 7,
                                  "multiple_shooting_penalty": 0.1},
    ("node", "multiple", True):  {"shooting_window_length": 5,
                                  "multiple_shooting_penalty": 0.1},
}


def prune_config(run_config: dict):
    # --- FILTER irrelevant params per conditions ---
    # if not run_config.get("use_dropout", False):
    #     run_config["input_dropout_rate"] = 0.0
    #     run_config["hidden_dropout_rate"] = 0.0

    # Handle training schemes
    run_config["curriculum_learning"] = "CL" in run_config.get("training_scheme")

    if "single" in run_config.get("training_scheme"):
        run_config["num_epochs_adam"] = N_EPOCHS_ADAM
        run_config["shooting_mode"] = "single"
        run_config["shooting_window_length"] = 0
    elif "multiple" in run_config.get("training_scheme"):
        run_config["num_epochs_adam"] = N_EPOCHS_ADAM
        run_config["shooting_mode"] = "multiple"
        if run_config["curriculum_learning"]:
            # in MS+CL the number of curriculum steps is dictated by the number of shooting windows -> set to
            # default value and ignore
            run_config["num_curriculum_steps"] = 5
    else:
        run_config["optimiser"] = "None"  # no NN optimiser
        run_config["num_epochs_adam"] = 0
        run_config["shooting_mode"] = "single"
        run_config["shooting_window_length"] = 0
        run_config["num_curriculum_steps"] = 0

    if "bfgs" in run_config.get("training_scheme"):
        run_config["num_epochs_bfgs"] = N_EPOCHS_BFGS
        run_config["shooting_mode"] = "single"  # set to single
        run_config["shooting_window_length"] = 0
    else:
        run_config["second_opt"] = "None"  # no second optimiser
        run_config["num_epochs_bfgs"] = 0

    if run_config.get("opt_sched_type") != "cos":
        # Only single schedules are supported for linear and exponential decays (+ warmup)
        run_config["opt_mult"] = 0
        run_config["opt_steps"] = 0

    if run_config.get("shooting_mode") == "single":
        run_config["shooting_iv_init"] = None
        run_config["multiple_shooting_penalty"] = 0.0
        run_config["shooting_window_length"] = 0  # set to 0, not used

    if not run_config.get("curriculum_learning", False):
        run_config["num_curriculum_steps"] = 0
        run_config["max_curriculum_epochs"] = 0
        run_config["restart_opt_curriculum"] = False

    if (not run_config.get("curriculum_learning", False)) and run_config.get("shooting_mode", None) == "multiple":
        run_config["num_curriculum_steps"] = 5  # set to default value, will be overwritten by number of windows

    # TODO - think this through - we might want to try width 5 without RBF or viceversa RBF with another width
    # # Replicate setup of UDE paper (arXiv link: https://arxiv.org/pdf/2001.04385, see page 40)
    # if (run_config.get("mlp_layer_width") == 5) and (run_config.get("mlp_depth") == 2):
    #     run_config["activation_fn_name"] = "rbf"
    #     run_config["max_lrate"] = 1e-1

    # Reaction-diffusion: the UDE/NODE variant is encoded in the model name, so force the matching
    # training_setting (redundant TRAINING_SETTINGS product entries are dropped by the dedup below).
    _dm = run_config.get("dynamical_model", "")
    if _dm.startswith("reaction_diffusion"):
        if "node" in _dm:
            run_config["training_setting"] = "NODE"
            run_config["output_reg"] = 0.0  # penalising a pure NODE's whole RHS is counter-productive
        elif "ude" in _dm:
            run_config["training_setting"] = "UDE"
            run_config["node_augmentation"] = 0  # augmentation is only meaningful for the NODE variant
        # Per-method hyperparameters, tuned on the 8x8 factorial scan. Applied last so these win
        # over the generic forcing above; keys absent from an entry keep the configured value.
        # Set RD_USE_METHOD_CONFIG = False in the config to scan these axes instead.
        if RD_USE_METHOD_CONFIG:
            _variant = "node" if "node" in _dm else "ude"
            _key = (_variant, run_config.get("shooting_mode"),
                    bool(run_config.get("curriculum_learning")))
            run_config.update(RD_METHOD_CONFIG.get(_key, {}))

    if run_config.get("training_setting") == "UDE":
        run_config["mlpexp_target"] = "None"
    elif run_config.get("training_setting") == "NODE":
        run_config["mlpexp_target"] = "None"
        # dynamic parameters are going to be frozen to zero, so they need to be evaluated linearly
        run_config["dynamic_params_scale"] = "linear"
        # and there is no need to sample widely around guess, since everything will be overridden with zeroes
        run_config["multistart_deviation"] = 0
    elif run_config.get("training_setting") == "MLP":
        run_config["output_reg"] = 0.0
        run_config["shooting_window_length"] = 0
        run_config["shooting_iv_init"] = "None"
        run_config["multiple_shooting_penalty"] = 0.0
        run_config["num_curriculum_steps"] = 0
        run_config["max_curriculum_epochs"] = 0
        run_config["dynamic_params_scale"] = "None"
        run_config["multistart_deviation"] = 0
        run_config["dynamic_params_perturb_method"] = "None"
        run_config["node_augmentation"] = 0
        run_config["second_opt"] = "None"
        params_set_to_false = [
            "curriculum_learning",
            "freeze_mechanistic_params_adam", "freeze_mechanistic_params_optx",
            "freeze_nn_adam", "freeze_nn_optx",
        ]
        for param_set_to_false in params_set_to_false:
            run_config[param_set_to_false] = False

    # # Only allow both normalisation options to be True or False -- removed, keep time/drop data normalisation
    # if not run_config.get("normalise_data"):
    #     run_config["normalise_timepoints"] = False
    # else:
    #     run_config["normalise_timepoints"] = True

    return run_config


# Opt-in full-factorial axes. By default every entry of linear_hyperparameters is scanned
# one-at-a-time (each value varied alone off its central value). A config may set
# FACTORIAL_SCAN_PARAMS to promote selected axes to product_hyperparameters, so that all of
# their combinations are enumerated instead -- needed when axes are expected to interact
# (e.g. lambda_cont x shooting window length). Absent/empty => unchanged one-at-a-time behaviour.
for _param in FACTORIAL_SCAN_PARAMS:
    if _param in linear_hyperparameters:
        product_hyperparameters[_param] = linear_hyperparameters.pop(_param)["range"]


def generate_param_combinations(n_starts: int) -> List[Dict]:
    # Check that all hyperparameter options are dicts (central value, range)
    if not all(
            isinstance(hyperparam, dict)
            for hyperparam in linear_hyperparameters.values()
    ):
        raise TypeError("Inconsistent typing for linear scans!")

    # Get central values
    central_values = {
        hyperparam: linear_hyperparameters[hyperparam]["central_value"]
        for hyperparam in scan_attributes
        if hyperparam in linear_hyperparameters
    }

    linear_scan_configs = [
        {**central_values, **{param: value}}
        for param in scan_attributes
        if param in linear_hyperparameters
        for value in linear_hyperparameters[param]["range"]
        if linear_hyperparameters[param]["central_value"] != value
    ] + [central_values]

    # product expand for starts, contexts, and features
    linear_scan_configs = [
        {**config, **{"dynamical_model": model, "multistart_job": start}}
        for config in linear_scan_configs
        for start, model in itt.product(
            range(n_starts), DYNAMICAL_MODEL
        )
    ]

    # product expand for product hyperparameters
    for param in product_hyperparameters:
        linear_scan_configs = [
            {**config, **{param: value}}
            for config in linear_scan_configs
            for value in product_hyperparameters[param]
        ]

    seen_configs = set()
    accepted_configs = []
    for run_config in linear_scan_configs:
        pruned_config = prune_config(run_config)
        # Create  hashable representation
        frozen_items = frozenset(sorted(pruned_config.items()))
        if frozen_items in seen_configs:
            continue  # skip duplicate
        seen_configs.add(frozen_items)
        accepted_configs.append(run_config)

    return accepted_configs

generate_param_combinations(
    n_starts=N_STARTS,
)
