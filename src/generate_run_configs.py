import itertools as itt

from src.model_training.training_configuration import (
    DYNAMICAL_MODEL,
    MLP_WIDTHS, MLP_DEPTHS,
    INIT_SCALES, USE_BIASES, USE_FINAL_BIASES,
    MLP_ACTIVATIONS,
    # MLP_FINAL_ACTIVATIONS,
    # USE_SKIP_CONNECTIONS,
    # USE_DROPOUT, INPUT_DROPOUT_RATES, HIDDEN_DROPOUT_RATES,
    OPTIMISERS,
    MAX_LEARNING_RATES, LEARNING_RATE_SPANS, LEARNING_RATE_DECAYS, WARMUP_FCTS, OPT_STEPS, OPT_MULT, OPT_SCHED_TYPES,
    # USE_LOOKAHEAD, LOOKAHEAD_SYNC,
    USE_EARLY_STOP,
    MLP_REG_EXPS, MLP_REG_STRENGTHS,
    OUTPUT_REG,
    SHOOTING_MODES, SHOOTING_WINDOW_LENGTHS, SHOOTING_IV_INITS, MULTIPLE_SHOOTING_PENALTIES,
    # BATCH_SIZES,
    CURRICULUM_LEARNING, NUM_CURRICULUM_STEPS, MAX_CURRICULUM_EPOCHS, RESTART_OPT_CURRICULUM,
    DYNAMIC_PARAMS_SCALES, MULTISTART_DEVIATIONS, DYNAMIC_PARAMS_PERTURB_METHODS,
    NODE_AUGMENTATIONS,
    HESSIAN_PRECONDITION, COMPUTE_HESSIAN_METRICS,
    SECOND_OPTS,
    # FREEZE_MECHANISTIC_PARAMS_ADAM, FREEZE_MECHANISTIC_PARAMS_OPTX, FREEZE_NN_ADAM, FREEZE_NN_OPTX,
    TRAINING_SETTINGS, MLPEXP_TARGETS,
    NORMALISE_DATA, NORMALISE_TIMEPOINTS,
    N_STARTS,
    N_EPOCHS_ADAM, N_EPOCHS_BFGS
)
from typing import List, Dict


def generate_param_combinations(n_starts: int) -> List[Dict]:
    # Build full param grid
    param_grid = {
        "dynamical_model": DYNAMICAL_MODEL,
        # MLP
        "mlp_layer_width": MLP_WIDTHS,
        "mlp_depth": MLP_DEPTHS,
        "init_scale": INIT_SCALES,
        "use_bias": USE_BIASES,
        "use_final_bias": USE_FINAL_BIASES,
        "activation_fn_name": MLP_ACTIVATIONS,
        # "final_activation_fn_name": MLP_FINAL_ACTIVATIONS,
        # Skip Connections
        # "use_skip_connections": USE_SKIP_CONNECTIONS,
        # Dropout
        # "use_dropout": USE_DROPOUT,
        # "input_dropout_rate": INPUT_DROPOUT_RATES,
        # "hidden_dropout_rate": HIDDEN_DROPOUT_RATES,
        # Optimiser
        "optimiser": OPTIMISERS,
        "num_epochs_adam": N_EPOCHS_ADAM,
        "max_lrate": MAX_LEARNING_RATES,
        "lrate_span": LEARNING_RATE_SPANS,
        "lrate_decay": LEARNING_RATE_DECAYS,
        "warmup_fct": WARMUP_FCTS,
        "opt_steps": OPT_STEPS,
        "opt_mult": OPT_MULT,
        "opt_sched_type": OPT_SCHED_TYPES,
        "use_early_stop": USE_EARLY_STOP,
        # Regularisation
        "mlp_reg_exp": MLP_REG_EXPS,
        "mlp_reg_strength": MLP_REG_STRENGTHS,
        "output_reg": OUTPUT_REG,
        # Shooting modes
        "shooting_mode": SHOOTING_MODES,
        "shooting_window_length": SHOOTING_WINDOW_LENGTHS,
        "shooting_iv_init": SHOOTING_IV_INITS,
        "multiple_shooting_penalty": MULTIPLE_SHOOTING_PENALTIES,
        # Curriculum learning
        "curriculum_learning": CURRICULUM_LEARNING,
        "num_curriculum_steps": NUM_CURRICULUM_STEPS,
        "max_curriculum_epochs": MAX_CURRICULUM_EPOCHS,
        "restart_opt_curriculum": RESTART_OPT_CURRICULUM,
        # Dynamic param settings
        "dynamic_params_scale": DYNAMIC_PARAMS_SCALES,
        "multistart_deviation": MULTISTART_DEVIATIONS,
        "multistart_job": list(range(n_starts)),
        "dynamic_params_perturb_method": DYNAMIC_PARAMS_PERTURB_METHODS,
        # Node augmentation
        "node_augmentation": NODE_AUGMENTATIONS,
        # Hessian
        "hessian_precondition": HESSIAN_PRECONDITION,
        "compute_hessian_metrics": COMPUTE_HESSIAN_METRICS,
        # Second optimiser
        "second_opt": SECOND_OPTS,
        "num_epochs_bfgs": N_EPOCHS_BFGS,
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
    }

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    seen_configs = set()
    all_param_configs = []

    for combination in itt.product(*values):
        params = dict(zip(keys, combination))

        # --- FILTER irrelevant params per conditions ---
        # if not params.get("use_dropout", False):
        #     params["input_dropout_rate"] = 0.0
        #     params["hidden_dropout_rate"] = 0.0

        if params.get("opt_sched_type") != "cos":
            # Only single schedules are supported for linear and exponential decays (+ warmup)
            params["opt_mult"] = 0
            params["opt_steps"] = 0

        if params.get("shooting_mode") == "single":
            params["shooting_iv_init"] = None
            params["multiple_shooting_penalty"] = 0.0
            params["shooting_window_length"] = 0  # set to 0, not used

        if not params.get("curriculum_learning", False):
            params["num_curriculum_steps"] = 0
            params["max_curriculum_epochs"] = 0
            params["restart_opt_curriculum"] = False

        if params.get("curriculum_learning", False) or (params.get("shooting_mode", None) != "single"):
            # With multiple shooting or curriculum learning, default to Adam and drop BFGS
            params["num_epochs_bfgs"] = 0
            params["optimiser"] = "adam"
            params["second_opt"] = "None"

        # Handle pure BFGS case
        if ((params.get("shooting_mode", None) == "single")
            and not params.get("curriculum_learning", False)
            and (params.get("optimiser", None) == "None")):
            params["num_epochs_adam"] = 0

        if params.get("curriculum_learning", False) and params.get("shooting_mode", None) == "multiple":
            params["num_curriculum_steps"] = 5  # set to default value, will be overwritten by number of windows

        # Replicate setup of UDE paper (arXiv link: https://arxiv.org/pdf/2001.04385, see page 40)
        if (params.get("mlp_layer_width") == 5) and (params.get("mlp_depth") == 2):
            params["activation_fn_name"] = "rbf"
            params["max_lrate"] = 1e-1

        if params.get("training_setting") == "UDE":
            params["mlpexp_target"] = "None"
        elif params.get("training_setting") == "MLP":
            params["output_reg"] = 0.0
            params["shooting_window_length"] = 0
            params["shooting_iv_init"] = "None"
            params["multiple_shooting_penalty"] = 0.0
            params["num_curriculum_steps"] = 0
            params["max_curriculum_epochs"] = 0
            params["dynamic_params_scale"] = "None"
            params["multistart_deviation"] = 0
            params["dynamic_params_perturb_method"] = "None"
            params["node_augmentation"] = 0
            params["second_opt"] = "None"
            params_set_to_false = [
                "curriculum_learning",
                "hessian_precondition",
                "freeze_mechanistic_params_adam", "freeze_mechanistic_params_optx",
                "freeze_nn_adam", "freeze_nn_optx",
            ]
            for param_set_to_false in params_set_to_false:
                params[param_set_to_false] = False

        # # Only allow both normalisation options to be True or False -- removed, keep time/drop data normalisation
        # if not params.get("normalise_data"):
        #     params["normalise_timepoints"] = False
        # else:
        #     params["normalise_timepoints"] = True

        # --- Create a hashable representation ---
        frozen_items = frozenset(sorted(params.items()))
        if frozen_items in seen_configs:
            continue  # skip duplicate
        seen_configs.add(frozen_items)
        all_param_configs.append(params)

    return all_param_configs

generate_param_combinations(
    n_starts=N_STARTS,
)
