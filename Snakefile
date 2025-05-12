from src.generate_run_configs import generate_param_combinations
from src.common import TRAINING_RESULTS, MERGED_TRAINING_RESULTS
from src.model_training.training_configuration import (
    DYNAMICAL_MODEL, N_STARTS, EXPERIMENT_NAME,
)

# Container if needed
singularity: "docker://giacomofabrini/test_ude:arm64"

param_combos = generate_param_combinations(N_STARTS)

rule train_udes:
    input:
        script = 'src/model_training/train.py'
    output:
        csv = TRAINING_RESULTS
    wildcard_constraints:
        dynamical_model =  "[a-zA-Z_]+",
        mlp_layer_width = "[0-9]+",
        mlp_depth = "[0-9]+",
        init_scale = "[0-9.]+",
        use_bias = "True|False",
        use_final_bias = "True|False",
        activation_fn_name = "[a-z]+",
        optimiser = "[a-z]+",
        max_lrate = "[0-9.]+",
        lrate_span = "[0-9.]+",
        lrate_decay = "[0-9.]+",
        warmup_fct = "[0-9.]+",
        opt_steps = "[0-9]+",
        opt_mult = "[0-9.]+",
        use_early_stop = "True|False",
        mlp_reg_exp = "[0-9.]+",
        mlp_reg_strength = "[0-9.]+",
        output_reg = "[0-9.]+",
        shooting_mode = "single|multiple",
        shooting_window_length = "[0-9]+",
        shooting_iv_init = "z|g|ru|None",
        multiple_shooting_penalty = "[0-9.]+",
        curriculum_learning = "True|False",
        num_curriculum_steps = "[0-9]+",
        max_curriculum_epochs = "[0-9]+",
        dynamic_params_scale = "linear|log10|None",
        multistart_deviation = "[0-9.]+",
        multistart_job = "[0-9]+",
        dynamic_params_perturb_method = "su|agn|None",
        node_augmentation = "[0-9]+",
        hessian_precondition = "True|False",
        compute_hessian_metrics = "True|False",
        second_opt = "fides|optx|None",
        normalise_data = "True|False",
        normalise_timepoints = "True|False",
        training_setting = "UDE|MLP",
        mlpexp_target = "[a-zA-Z_]+",
        experiment_name = "[0-9a-zA-Z_]+"
    retries: 1
    resources:
        mem="4GB",
        runtime="24h",
        nodes=1,
        threads=2,
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in (
                "dynamical_model",
                "mlp_layer_width", "mlp_depth",
                "init_scale",
                "use_bias", "use_final_bias",
                "activation_fn_name",
                "optimiser", "max_lrate", "lrate_span", "lrate_decay", "warmup_fct", "opt_steps", "opt_mult",
                "use_early_stop",
                "mlp_reg_exp", "mlp_reg_strength",
                "output_reg",
                "shooting_mode", "shooting_window_length", "shooting_iv_init",
                "multiple_shooting_penalty",
                "curriculum_learning", "num_curriculum_steps", "max_curriculum_epochs",
                "dynamic_params_scale",
                "multistart_deviation", "multistart_job",
                "dynamic_params_perturb_method",
                "node_augmentation",
                "hessian_precondition", "compute_hessian_metrics",
                "second_opt",
                "normalise_data", "normalise_timepoints",
                "training_setting", "mlpexp_target",
                "experiment_name"
            )
        )

# training_output = expand(
#     TRAINING_RESULTS,
#     dynamical_model = DYNAMICAL_MODEL,
#     mlp_layer_width=MLP_WIDTHS, mlp_depth=MLP_DEPTHS,
#     init_scale=INIT_SCALES,
#     use_bias=USE_BIASES, use_final_bias=USE_FINAL_BIASES,
#     activation_fn_name=MLP_ACTIVATIONS,
#     optimiser=OPTIMISERS,
#     max_lrate=MAX_LEARNING_RATES, lrate_span=LEARNING_RATE_SPANS, lrate_decay=LEARNING_RATE_DECAYS,
#     warmup_fct=WARMUP_FCTS, opt_steps=OPT_STEPS, opt_mult=OPT_MULT,
#     use_early_stop=USE_EARLY_STOP,
#     mlp_reg_exp=MLP_REG_EXPS, mlp_reg_strength=MLP_REG_STRENGTHS,
#     output_reg=OUTPUT_REG,
#     shooting_mode=SHOOTING_MODES,
#     shooting_window_length=SHOOTING_WINDOW_LENGTHS,
#     shooting_iv_init=SHOOTING_IV_INITS,
#     multiple_shooting_penalty=MULTIPLE_SHOOTING_PENALTIES,
#     curriculum_learning=CURRICULUM_LEARNING,
#     num_curriculum_steps=NUM_CURRICULUM_STEPS,
#     max_curriculum_epochs=MAX_CURRICULUM_EPOCHS,
#     dynamic_params_scale=DYNAMIC_PARAMS_SCALES,
#     multistart_deviation=MULTISTART_DEVIATIONS,
#     multistart_job=[str(i) for i in range(N_STARTS)],
#     dynamic_params_perturb_method=DYNAMIC_PARAMS_PERTURB_METHODS,
#     node_augmentation=NODE_AUGMENTATIONS,
#     hessian_precondition=HESSIAN_PRECONDITION, compute_hessian_metrics=COMPUTE_HESSIAN_METRICS,
#     second_opt=SECOND_OPTS,
#     normalise_data=NORMALISE_DATA, normalise_timepoints=NORMALISE_TIMEPOINTS,
#     training_setting=TRAINING_SETTINGS, mlpexp_target=MLPEXP_TARGETS,
#     experiment_name=EXPERIMENT_NAME
# )

# print(f"Training output: {training_output}")
#
# new_training_output = [
#     y
#     for hyperparam_config in param_combos
#     for y in expand(
#         rules.train_udes.output.csv,
#         **hyperparam_config,
#         experiment_name=EXPERIMENT_NAME,
#     )
# ]
# print(f"New training output: {new_training_output}")
rule merge_csvs:
    input:
        script = 'src/merge_csvs.py',
        training = [
            y
            for hyperparam_config in param_combos
            for y in expand(
                rules.train_udes.output.csv,
                **hyperparam_config,
                experiment_name=EXPERIMENT_NAME,
            )
        ]
    output:
        expand(
            MERGED_TRAINING_RESULTS,
            dynamical_model=DYNAMICAL_MODEL,
            experiment_name=EXPERIMENT_NAME
        )
    shell:
        "python3 {input.script}"


rule all:
    input:
        rules.merge_csvs.output
