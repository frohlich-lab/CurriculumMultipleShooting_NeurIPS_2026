import itertools as itt
import math
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

# Per-method reaction-diffusion hyperparameters.
#
# Continuity penalty: plain MS keeps the configured value (MULTIPLE_SHOOTING_PENALTIES = 1.0) while
# CMS runs WITHOUT one, the curriculum alone tying the windows together.
#
# Learning rate: tuned per method by the 08/08/2026 sweep (experiment
# reaction_diffusion_lrsweep_08082026; 2 variants x 4 schemes x 5 rates x 5 seeds, 500 epochs).
# Each method gets its own rate deliberately -- comparing methods at a shared rate handicaps
# whichever ones disagree with it, and the optima here span 1e-4 to 1e-2. Selection was on median
# TRAINING mse only, at a matched budget (epoch <= 400) so the 31 wall-clock-truncated runs could
# be included from their W&B history; validation/forecast MSE and parameter recovery were held out
# of the choice. The ranking is stable: 7 of 8 cells pick the same rate under the full 500-epoch
# budget on completed runs only.
#
# Caveats recorded with the numbers they came from:
#   * ude/single -- that row is nearly flat (0.817 / 0.843 / 0.845 at 1e-4 / 3e-4 / 1e-3), and the
#     full-budget view prefers 3e-4. Treat as "anything <= 1e-3", not a sharp optimum.
#   * node/multiple -- BROKEN at every rate (median mse_train 2.9e7 ... 2.44, all at or above the
#     predict-zeros baseline of 1.0032). 1e-2 is merely least-bad; this arm needs a fix, not a
#     learning rate, and 1e-2 also sits in the regime where the first Adam step produces gradients
#     up to 1e132 (caught by the clip, see make_grads_finite).
#
# Continuity penalty: pinned per method from the penalty scan (experiment
# reaction_diffusion_geomscan_21082026, penscan array 54672962, completed 09/2026). Selection on
# median TRAINING mse at L=8 with output_reg at its central value (1e4 UDE / 0 NODE), 10 seeds,
# matched-seed comparison over pen in {0, 0.01, 0.1, 1}. This supersedes the 5-seed regscan pins.
# This table is applied LAST in prune_config, so a pin here OVERRIDES any scan over
# MULTIPLE_SHOOTING_PENALTIES -- set RD_USE_METHOD_CONFIG = False to scan the axis again.
#
# Median train mse per pen (0 / 0.01 / 0.1 / 1) and the evidence for each pin:
#   ude/multiple      0.1  -- 0.4619 / 0.4449 / 0.3341 / 0.4760. 8 of 10 seeds prefer 0.1, margin
#                             1.33x. Resolves the 5-seed median-vs-vote split (was pinned 0.0).
#   ude/multiple_CL   0.0  -- 0.1572 / 0.4372 / 0.3565 / 0.2504. Margin 1.59x, votes 6/4. Confirmed.
#   node/multiple     1.0  -- 4.9312 / 5.3708 / 3.6499 / 3.0621. Margin 1.19x, but this arm never
#                             beats the predict-zeros baseline (1.0032) at any penalty, so the value
#                             is cosmetic: it ranks degrees of failure. See the max_lrate note above.
#   node/multiple_CL  0.01 -- 0.0602 / 0.0435 / 0.0543 / 0.0718. Margin 1.25x, votes spread
#                             4/3/1/2 -- the weakest of the four (was pinned 0.0).
# The axis is non-monotone in three of four arms, so treat these as best-of-grid, not a trend.
#
# CL learning rates re-checked in the lrcl scan (same experiment, array 58754585, 10 seeds, two
# half-decade steps below each pin and one above): all four pins stand. ude/single_CL is now
# bracketed (1e-5 0.815 / 3e-5 0.327 / 1e-4 0.171 / 3e-4 0.235); node/multiple_CL wins on 10/10
# seeds. ude/multiple_CL at 1e-3 has a lower median (0.136 vs 0.157) but beats the pin on only 5/10
# seeds, and was a cliff in the 08/08 sweep, so 3e-4 is kept.
#
# Non-CL learning rates re-checked in the lrnoncl scan (array 59517215, same design, 10 seeds;
# median train mse at 2 below / 1 below / pin / 1 above):
#   ude/single    1e-4 kept -- 1.295 / 0.920 / 0.809 / 0.818. A tie: 7/10 seeds marginally prefer
#                              3e-4 while the median prefers 1e-4 (1.01x). The arm barely learns
#                              (baseline 1.003), so the choice is immaterial; not moved on a split.
#   ude/multiple  1e-3 kept -- 0.416 / 0.338 / 0.334 / 0.792 at penalty 0.1; no rate beats the pin
#                              on more than 3/10 seeds.
#   node/single   1e-4 -> 3e-4 -- 1.059 / 0.825 / 0.367 / 0.273; 9/10 seeds better, 1.35x. 1e-3 was
#                              0.635 in the 08/08 sweep, so the optimum is bracketed.
#
# single_CL curriculum stages: 12, decoupled from the window length (stages scan, array 58754586,
# 10 seeds, pinned lr). Median train mse at 5 / 7 / 12 / 18 stages:
#   ude/single_CL   0.171 / 0.155 / 0.130 / 0.132   (8/10 seeds better at 12 than 5)
#   node/single_CL  0.065 / 0.053 / 0.049 / 0.054   (7/10 seeds better at 12 than 5)
# 12 vs 18 is a near-tie, so this is "the curve has flattened", not a sharp optimum. This
# deliberately BREAKS the stage-count match with multiple_CL (5 windows at L=8) that prune_config
# otherwise imposes: each method gets its own tuned curriculum, as with the learning rates.
# max_curriculum_epochs co-varies for equal stages (see curriculum_from_windows).
RD_SINGLE_CL_STAGES = 12
_RD_SINGLE_CL_CURRICULUM = {
    "num_curriculum_steps": RD_SINGLE_CL_STAGES,
    "max_curriculum_epochs": int(round(N_EPOCHS_ADAM * (RD_SINGLE_CL_STAGES - 1) / RD_SINGLE_CL_STAGES)),
}

# Keys are (variant, shooting_mode, curriculum_learning).
RD_METHOD_CONFIG = {
    ("ude",  "single",   False): {"max_lrate": 1e-4},
    ("ude",  "single",   True):  {"max_lrate": 1e-4, **_RD_SINGLE_CL_CURRICULUM},
    ("ude",  "multiple", False): {"max_lrate": 1e-3, "multiple_shooting_penalty": 0.1},
    ("ude",  "multiple", True):  {"max_lrate": 3e-4, "multiple_shooting_penalty": 0.0},
    ("node", "single",   False): {"max_lrate": 3e-4},
    ("node", "single",   True):  {"max_lrate": 1e-3, **_RD_SINGLE_CL_CURRICULUM},
    ("node", "multiple", False): {"max_lrate": 1e-2, "multiple_shooting_penalty": 1.0},
    ("node", "multiple", True):  {"max_lrate": 1e-3, "multiple_shooting_penalty": 0.01},
}


def num_shooting_windows(run_config: dict):
    """Number of multiple-shooting windows this config will actually produce, or None if unknown.

    Mirrors initialize_multiple_shooting / setup_curriculum_learning exactly: windows overlap by one
    point, so each advances by (L-1) and they must cover the (n-1) intervals between n timepoints.
    Only the reaction-diffusion models have a training length known at config-generation time.
    """
    model = run_config.get("dynamical_model", "")
    if not model.startswith("reaction_diffusion"):
        return None
    length = run_config.get("shooting_window_length") or 0
    if length <= 1:
        return None
    from src.model_examples.reaction_diffusion.setup_reaction_diffusion_model import N_TRAIN_POINTS
    return int(math.ceil((N_TRAIN_POINTS - 1) / (length - 1)))


def curriculum_from_windows(run_config: dict, n_win):
    """Set the curriculum stage count to `n_win` and co-vary max_curriculum_epochs to match.

    setup_curriculum_learning spreads the stage STARTS over [0, max_curriculum_epochs] as
    ``interval = max_curriculum_epochs / (n_stages - 1)``, and the final full-data stage then runs
    from there to num_epochs_adam. Equal-length stages therefore require

        max_curriculum_epochs = num_epochs_adam * (n_stages - 1) / n_stages

    so any scan over shooting_window_length silently unbalances the curriculum unless this tracks it.
    """
    run_config["num_curriculum_steps"] = n_win
    run_config["max_curriculum_epochs"] = int(
        round(run_config["num_epochs_adam"] * (n_win - 1) / n_win)
    ) if n_win > 1 else run_config["num_epochs_adam"]


def prune_config(run_config: dict):
    # --- FILTER irrelevant params per conditions ---
    # if not run_config.get("use_dropout", False):
    #     run_config["input_dropout_rate"] = 0.0
    #     run_config["hidden_dropout_rate"] = 0.0

    # Handle training schemes
    run_config["curriculum_learning"] = "CL" in run_config.get("training_scheme")

    # Read the SCANNED window length before the single-shooting branch below zeroes it. Both CL arms
    # need it: for multiple+CL the window count is literally the stage count, and single+CL must use
    # the SAME stage count or the two stop being comparable -- NUM_CURRICULUM_STEPS was originally
    # pinned to 5 precisely because that is the window count at the default length of 8. Scanning the
    # window length therefore rescales the curriculum granularity for single+CL too, even though
    # single shooting has no windows of its own.
    # For reaction-diffusion with RD_USE_METHOD_CONFIG this match is deliberately overridden:
    # single+CL is pinned to RD_SINGLE_CL_STAGES (tuned separately) in RD_METHOD_CONFIG.
    _n_win_scanned = num_shooting_windows(run_config)

    if "single" in run_config.get("training_scheme"):
        run_config["num_epochs_adam"] = N_EPOCHS_ADAM
        run_config["shooting_mode"] = "single"
        if run_config["curriculum_learning"] and _n_win_scanned is not None:
            curriculum_from_windows(run_config, _n_win_scanned)
        run_config["shooting_window_length"] = 0
    elif "multiple" in run_config.get("training_scheme"):
        run_config["num_epochs_adam"] = N_EPOCHS_ADAM
        run_config["shooting_mode"] = "multiple"
        if run_config["curriculum_learning"]:
            if _n_win_scanned is not None:
                curriculum_from_windows(run_config, _n_win_scanned)
            else:
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
