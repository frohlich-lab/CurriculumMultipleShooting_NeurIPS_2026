# Reaction-diffusion (FitzHugh-Nagumo) benchmark, adapted from APHYNITY (arXiv:2010.04456).
# Two variants: hybrid UDE (known diffusion + ConvNet reaction) and vanilla augmented NODE.
#
# CURRENT MODE: final benchmark run (reaction_diffusion_final_19102026), seeds 10..109.
#   - every axis is a single value; per-method learning rate, continuity penalty and single_CL
#     stage count come from RD_METHOD_CONFIG in generate_run_configs.py (evidence recorded there)
#   - output_reg 1e4 (UDE only -- NODE forces 0), shooting window length 8 (5 windows)
#   - 500 Adam epochs; equal-length curriculum stages (max_curriculum_epochs co-varies with the
#     stage count, see prune_config)
#   - both channels observed (u+v) -- set RD_OBSERVED=uv in the launcher; do NOT run the u-only arm
#
# To activate: copy this file over src/model_training/training_configuration.py.

DYNAMICAL_MODEL = {
    "reaction_diffusion_ude_64",
    "reaction_diffusion_node_64",
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
    # SUPERSEDED per method: RD_USE_METHOD_CONFIG routes each (variant, shooting_mode,
    # curriculum_learning) to its own tuned rate via RD_METHOD_CONFIG in generate_run_configs.py.
    # This value is only the fallback for methods absent from that table, so it must stay a single
    # entry -- a multi-valued range here would expand configs that RD_METHOD_CONFIG then collapses
    # back to duplicates.
    #
    # Set RD_USE_METHOD_CONFIG = False to scan this axis again instead.
    #
    # On 1e-2: the sweep confirmed it destabilises on the FIRST Adam step -- every one of the 32
    # runs with a raw gradient above 1e10 was at 3e-3 or 1e-2, and every one peaked at epoch 1.
    # The earlier note recording this was measured before the solver step-size cap went in, so it
    # conflated two effects, but its conclusion was right: 1e-2 is worst or near-worst in 7 of 8
    # (variant, scheme) cells.
    "range": (1e-3,),
    "central_value": 1e-3,
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
    # LINEAR scan (one-at-a-time off the central value). Output regularisation on the NN (reaction)
    # output; curbs the ConvNet from absorbing the known diffusion, so it is the lever on (a, b)
    # identifiability. UDE only -- prune_config forces 0 for NODE, so the NODE arms dedup to a single
    # config and this axis costs nothing there.
    #
    # RANGE RESCALED BY 3 ORDERS OF MAGNITUDE (20/08/2026). The old range (0, 1, 10, 100) was inert:
    # the 12-19/08 regscan measured a median train-mse spread of 1.00-1.07x across it, and in UDE
    # single the diffusion estimates were bit-identical to 6 significant figures at 0 / 1 / 100.
    # Cause is the 1e-6 pre-multiplier in regularisation.py:40 --
    # `1e-6 * reg_strength * (integral/timespan)**2` -- which maps 0-100 onto effective coefficients
    # of 0-1e-4, i.e. 2e-6 to 4e-3 of the objective. The old default of 10 did nothing at all.
    #
    # Calibrated by the oregprobe (experiment reaction_diffusion_oregprobe_19082026, UDE single_CL,
    # 4 values x 5 seeds, all 20 runs complete). Medians over 5 seeds at CONVERGENCE (epoch 499):
    #     output_reg=10    reg/loss  0.0%   mse_train 0.168   1.00x   (the old default: inert)
    #     output_reg=1e4   reg/loss 19.2%   mse_train 0.179   1.07x   <- new central value
    #     output_reg=1e5   reg/loss 30.0%   mse_train 0.310   1.85x
    #     output_reg=1e6   reg/loss 11.6%   mse_train 0.724   4.32x   <- DROPPED, over-regularises
    #
    # Why 1e4 is the central value: it is the point where the penalty is already a SUBSTANTIAL share
    # of the objective (19%) while costing almost nothing in fit (+7%). 1e5 buys more penalty (30%)
    # for a large fit cost (+85%); 1e6 overwhelms the data term entirely.
    #
    # reg/loss is NON-MONOTONE in output_reg at convergence (0 -> 19.2% -> 30.0% -> 11.6%, peaking at
    # 1e5). Above ~1e5 the penalty clamps ||NN|| so hard that the term itself shrinks -- the signature
    # of it genuinely binding. At 1e6 that also collapses the between-seed spread of mse_train to
    # 0.700-0.784 (vs a 6x spread at the control): every run converges to the same over-regularised
    # solution regardless of initialisation.
    #
    # CAUTION on predicting 1e3 (the untested low anchor): the reg fraction is strongly
    # seed-dependent, not a clean function of output_reg -- at 1e4 it spans 0.3%-24.6% across the five
    # seeds. Earlier mid-training readings (3.2% at 1e4) also badly understated the converged value,
    # so do not extrapolate; measure it. 1e3 replaces the old inert values, as there is no point
    # spending seeds on 0/1/10/100 again.
    #
    # NOTE the aggregate hides the per-run picture: only 2 of 5 seeds are individually monotone in
    # output_reg. Seed 0 IMPROVES with regularisation (0.774 -> 0.600 at 1e5) and seed 4 degrades
    # badly then recovers (0.653 -> 1.449 -> 0.700). The clean monotone median is a median, not a
    # law -- expect per-seed reversals.
    #
    # IMPORTANT -- this axis is NOT tunable on the selection metric. Median train mse rises
    # monotonically with output_reg (0.168 -> 0.179 -> 0.310 -> 0.724), so selecting on train mse always
    # drives it to zero, which is exactly the setting the identifiability argument rejects. The
    # central value of 1e4 is therefore chosen A PRIORI as the largest value whose fit cost is small
    # (~7%), not tuned. Deciding it on evidence would require the (a, b) recovery numbers, which are
    # deliberately held out during tuning.
    #
    # FINAL (10/2026): collapsed to the single policy value 1e4 for the benchmark run. The 1e3 / 1e5
    # arms of the geomscan remain in reaction_diffusion_geomscan_21082026 if the axis is revisited.
    "range": (1e4,),
    "central_value": 1e4,
}

# Shooting parameters (shooting mode is decided by TRAINING_SCHEMES)
SHOOTING_WINDOW_LENGTHS = {
    # LINEAR scan: varied with multiple_shooting_penalty held at its (now per-method) value.
    #
    # 36 train points, windows overlapping by one point -> n_windows = ceil(35/(L-1)):
    #     L=4 -> 12 windows    L=6 -> 7    L=8 -> 5 (central)    L=13 -> 3
    #
    # L=4 and L=13 DROPPED after the 12-19/08/2026 regscan:
    #   L=13 (3 windows) was the worst setting in 5 of 6 cells on median train RMS (+21% to +47%),
    #        so it is simply too few windows -- no reason to keep paying for it.
    #   L=4 (12 windows) was competitive on accuracy (best for NODE multiple_CL, +14% for UDE
    #        multiple_CL) but is not worth its cost: 2.5-4.0x the runtime of L=8 for gains of 7-19%
    #        that sit inside the seed-to-seed spread. It is also the only setting that did not fit
    #        the wall: both regscan TIMEOUTs were UDE multiple_CL at L=4, needing ~174h at 8 cores,
    #        beyond even the 7-day ncpu cap. Reinstating it requires 16 cores (~124h).
    #
    # Single-shooting arms force this to 0, so they dedup to one config per cell.
    # NOTE: for multiple+CL the window count IS the curriculum stage count, and prune_config
    # co-varies max_curriculum_epochs to keep the stages equal. Dropping L=4 therefore also removes
    # the 12-stage curriculum from BOTH CL arms -- including single_CL, which has no windows and so
    # pays none of the cost above (see NUM_CURRICULUM_STEPS).
    #
    # FINAL (10/2026): L=8 for every MS arm. Geomscan, matched 10 seeds, median train mse L=6 / L=8:
    #   ude/multiple 0.893 / 0.462 (0/10 seeds prefer 6)   ude/multiple_CL 0.330 / 0.157 (4/10)
    #   node/multiple 3.574 / 3.062 (3/10, arm never learns) node/multiple_CL 0.046 / 0.060 (6/10)
    # The one dissent, node/multiple_CL, was measured at penalty 0; at its new pin 0.01, L=8 reaches
    # 0.0435, already below L=6's 0.046 (L=6 at 0.01 itself was not run).
    "range": (8,),
    "central_value": 8,
}

SHOOTING_IV_INITS = {
    "z",                 # zeros (+ eps); first window = u0 guess
}

MULTIPLE_SHOOTING_PENALTIES = {
    # SUPERSEDED per method: RD_USE_METHOD_CONFIG routes each (variant, shooting_mode,
    # curriculum_learning) to its own penalty via RD_METHOD_CONFIG in generate_run_configs.py, which
    # records the per-arm evidence and its (uneven) confidence. Like MAX_LEARNING_RATES, this must
    # stay a SINGLE entry -- a multi-valued range here would expand configs that RD_METHOD_CONFIG
    # then collapses back to duplicates.
    #
    # Set RD_USE_METHOD_CONFIG = False to scan this axis again. If you do, scan (0.0, 0.01, 0.1, 1.0):
    # the regscan showed 10.0 is worst in every arm and 3 of 4 arms optimise at <= 0.1, so the
    # unresolved region is BELOW 1.0, not above it.
    "range": (0.0,),
    "central_value": 0.0,
}

# Curriculum learning (decided by TRAINING_SCHEMES)
NUM_CURRICULUM_STEPS = {
    # 5 = window count at L=8. For reaction diffusion single_CL is OVERRIDDEN to 12 stages
    # (RD_SINGLE_CL_STAGES in generate_run_configs.py, tuned in the stages scan); this value only
    # applies when RD_USE_METHOD_CONFIG is False.
    "range": (5,),
    "central_value": 5,
}

MAX_CURRICULUM_EPOCHS = {
    # 80% of the 500-epoch Adam budget, chosen so all five curriculum stages get an equal share.
    #
    # setup_curriculum_learning spreads the stage STARTS over [0, max_curriculum_epochs] as
    # `interval = max_curriculum_epochs / (num_curriculum_steps - 1)`, and the final full-data stage
    # then runs from max_curriculum_epochs to N_EPOCHS_ADAM. With 5 stages:
    #     interval = 400/4 = 100  ->  stages start at 0, 100, 200, 300, 400
    #     stages 1-4: 100 epochs each;  stage 5 (full data): 400 -> 500, also 100
    # i.e. 20% each. Setting this to 500 instead would give the first four stages 125 epochs and the
    # full-data stage only the final epoch.
    #
    # Applies to single+CL, where num_curriculum_steps comes from this config (5). For multiple+CL
    # the stage count IS the window count, so prune_config DERIVES both num_curriculum_steps and
    # max_curriculum_epochs from shooting_window_length and ignores this value -- otherwise a scan
    # over window length would silently unbalance the curriculum.
    # Keep at 0.8 * N_EPOCHS_ADAM if the epoch budget changes.
    "range": (400,),
    "central_value": 400,
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
    # +-1 order of magnitude. Was 2.0, but a landscape scan over Du shows the single-shooting
    # objective is flat (and above the predict-zero baseline) beyond ~0.4x truth and has a
    # REVERSED gradient beyond ~3x truth, so starting +-2 decades out puts single shooting where
    # descent moves away from the solution. Multiple shooting stays monotone over the same range.
    1.0,
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

# 110 for the final confirmation run, which uses seeds 10..109 only. Every tuning scan ran seeds
# 0..9, and multistart_job also seeds the NN initialisation (train.py), so reusing 0..9 would rerun
# the exact inits the settings were selected on (for NODE, which samples no (a, b), literally the
# same runs). 110 keeps 100 untouched seeds with no change to the multistart loader. The (a, b) LH
# sample at n=110 is a fresh draw (src/data_dir/<model>/lh__*__110.csv).
N_STARTS = 110  # number of multi-starts (seeds)

N_EPOCHS_ADAM = 500    # learning-rate sweep budget (was 3000 for the full spiral benchmark)
N_EPOCHS_BFGS = 1000  # unused (single_bfgs removed from TRAINING_SCHEMES)

EXPERIMENT_NAME = "reaction_diffusion_final_19102026"

# ON: used only to give CMS a zero continuity penalty (see RD_METHOD_CONFIG in
# src/generate_run_configs.py). All other hyperparameters come from the single global values
# above -- the 8x8 APHYNITY tuning of window length / output_reg does NOT transfer to this
# dataset and is no longer applied.
# No factorial axes: every scanned axis is varied ONE AT A TIME off its central value.
# Caveat to keep in mind when reading the results: shooting_window_length and
# multiple_shooting_penalty interact (more windows -> more discontinuity terms, so the penalty that
# balances the data term shifts with the window count). A linear scan varies each with the other held
# at its central value, so it measures their main effects only and cannot see that interaction.
# Promote both into this set to switch to a full factorial.
FACTORIAL_SCAN_PARAMS = frozenset()

RD_USE_METHOD_CONFIG = True
