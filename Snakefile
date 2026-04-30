# import itertools as itt

from src.generate_run_configs import generate_param_combinations
from src.common import (
    data_dir,
    reformat_filename,
    # AGGREGATION_PLOTS,
    OUTPUT_HDF5,
    # PLOT_TYPES,
    TRAINING_RESULTS
)
from src.model_training.training_configuration import (
    DYNAMICAL_MODEL,
    N_STARTS,
    EXPERIMENT_NAME,
    # MLP_ACTIVATIONS,
    # MAX_LEARNING_RATES,
    # MULTIPLE_SHOOTING_PENALTIES,
)

# Container if needed
singularity: "docker://giacomofabrini/test_ude:amd64"

param_combos = generate_param_combinations(N_STARTS)

# Get unique combinations for multistart generation
unique_multistart_settings = sorted(set(
    (p["dynamical_model"], p["multistart_deviation"], p["dynamic_params_perturb_method"])
    for p in param_combos
))

# Expand to file paths
multistart_csvs = [
    data_dir / f"{model}/{method}__{deviation}__{N_STARTS}.csv"
    for model, deviation, method in unique_multistart_settings
]


rule generate_multistarts:
    input:
        script="src/generate_multistarts.py"
    output:
        multistart_csvs
    resources:
        mem="2GB",
        runtime="1h",
        threads=1
    shell:
        "python3 {input.script}"


rule train_udes:
    input:
        script = 'src/model_training/train.py',
        multistarts=rules.generate_multistarts.output
    output:
        csv = reformat_filename(TRAINING_RESULTS)
    wildcard_constraints:
        dynamical_model =  "[a-zA-Z_]+",
        mlp_layer_width = "[0-9]+",
        mlp_depth = "[0-9]+",
        activation_fn_name = "[a-z]+",
        training_scheme = "bfgs|single|single_bfgs|single_CL|multiple|multiple_CL",
        optimiser = "[a-z]+|None",
        num_epochs_adam = "[0-9]+",
        max_lrate = "[0-9.]+",
        lrate_decay = "[0-9.]+",
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
        dynamic_params_perturb_method = "lh|su|agn|None",
        node_augmentation = "[0-9]+",
        compute_hessian_metrics = "True|False",
        second_opt = "fides|optx|None",
        num_epochs_bfgs = "[0-9]+",
        normalise_data = "True|False",
        normalise_timepoints = "True|False",
        training_setting = "UDE|NODE|MLP",
        mlpexp_target = "[a-zA-Z_]+",
        save_landscape_models = "True|False",
        noise_level = "[0-9]+|None",
        time_span = "[0-9]+|None",
    retries: 1
    resources:
        mem="32GB",
        runtime="24h",
        nodes=1,
        threads=2
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in (
                "dynamical_model",
                "mlp_layer_width",
                "mlp_depth",
                "activation_fn_name",
                "training_scheme",
                "optimiser",
                "num_epochs_adam",
                "max_lrate",
                "lrate_decay",
                "mlp_reg_exp",
                "mlp_reg_strength",
                "output_reg",
                "shooting_mode",
                "shooting_window_length",
                "shooting_iv_init",
                "multiple_shooting_penalty",
                "curriculum_learning",
                "num_curriculum_steps",
                "max_curriculum_epochs",
                "dynamic_params_scale",
                "multistart_deviation",
                "multistart_job",
                "dynamic_params_perturb_method",
                "node_augmentation",
                "compute_hessian_metrics",
                "second_opt",
                "num_epochs_bfgs",
                "normalise_data",
                "normalise_timepoints",
                "training_setting",
                "mlpexp_target",
                "save_landscape_models",
                "noise_level",
                "time_span",
            )
        ) + ' --experiment_name={EXPERIMENT_NAME} --num_starts={N_STARTS}'


rule aggregate_results:
    input:
        script = 'src/aggregate_results.py',
        training = [
            reformat_filename(y)
            for hyperparam_config in param_combos
            for y in expand(
                rules.train_udes.output.csv,
                **hyperparam_config,
                experiment_name=EXPERIMENT_NAME,
            )
        ]
    output:
        expand(
            OUTPUT_HDF5,
            experiment_name=EXPERIMENT_NAME,
            dynamical_model=DYNAMICAL_MODEL,
        )
        # expand(
        #     AGGREGATION_PLOTS,
        #     experiment_name=EXPERIMENT_NAME,
        #     dynamical_model=DYNAMICAL_MODEL,
        #     filename=[
        #         f"{plot_type}__{activation_fn_name}__{max_lrate}__{multiple_shooting_penalty}"
        #         for plot_type in PLOT_TYPES
        #         for activation_fn_name, max_lrate, multiple_shooting_penalty in itt.product(
        #             MLP_ACTIVATIONS,
        #             MAX_LEARNING_RATES,
        #             MULTIPLE_SHOOTING_PENALTIES
        #         )
        #     ]
        # )
    retries: 1
    resources:
        mem="64GB",
        runtime="6h",
        nodes=1
    shell:
        "python3 {input.script}"


rule all:
    input:
        rules.aggregate_results.output
