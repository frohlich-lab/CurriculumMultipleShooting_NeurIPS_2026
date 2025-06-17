import itertools as itt

from src.generate_run_configs import generate_param_combinations
from src.common import TRAINING_RESULTS, AGGREGATION_PLOTS, PLOT_TYPES
from src.model_training.training_configuration import (
    DYNAMICAL_MODEL, N_STARTS, EXPERIMENT_NAME, MLP_ACTIVATIONS, MAX_LEARNING_RATES,
)

# Container if needed
singularity: "docker://giacomofabrini/test_ude:amd64"

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
        optimiser = "[a-z]+|None",
        max_lrate = "[0-9.]+",
        lrate_span = "[0-9.]+",
        lrate_decay = "[0-9.]+",
        warmup_fct = "[0-9.]+",
        opt_steps = "[0-9]+",
        opt_mult = "[0-9.]+",
        opt_sched_type = "lin|exp|cos",
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
        restart_opt_curriculum = "True|False",
        dynamic_params_scale = "linear|log10|None",
        multistart_deviation = "[0-9.]+",
        multistart_job = "[0-9]+",
        dynamic_params_perturb_method = "lh|su|agn|None",
        node_augmentation = "[0-9]+",
        compute_hessian_metrics = "True|False",
        second_opt = "fides|optx|None",
        normalise_data = "True|False",
        normalise_timepoints = "True|False",
        training_setting = "UDE|MLP",
        mlpexp_target = "[a-zA-Z_]+",
        experiment_name = "[0-9a-zA-Z_]+"
    retries: 1
    resources:
        mem="8GB",
        runtime="6h",
        nodes=1,
        threads=2
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in (
                "dynamical_model",
                "mlp_layer_width",
                "mlp_depth",
                "init_scale",
                "use_bias",
                "use_final_bias",
                "activation_fn_name",
                "optimiser",
                "max_lrate",
                "lrate_span",
                "lrate_decay",
                "warmup_fct",
                "opt_steps",
                "opt_mult",
                "opt_sched_type",
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
                "restart_opt_curriculum",
                "dynamic_params_scale",
                "multistart_deviation",
                "multistart_job",
                "dynamic_params_perturb_method",
                "node_augmentation",
                "compute_hessian_metrics",
                "second_opt",
                "normalise_data",
                "normalise_timepoints",
                "training_setting",
                "mlpexp_target",
                "experiment_name"
            )
        )


rule aggregate_results:
    input:
        script = 'src/aggregate_results.py',
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
            AGGREGATION_PLOTS,
            experiment_name=EXPERIMENT_NAME,
            dynamical_model=DYNAMICAL_MODEL,
            filename=[
                f"{plot_type}__{activation_fn_name}__{max_lrate}"
                for plot_type in PLOT_TYPES
                for activation_fn_name, max_lrate in itt.product(
                    MLP_ACTIVATIONS,
                    MAX_LEARNING_RATES
                )
            ]
        )
    retries: 1
    resources:
        mem="32GB",
        runtime="3h",
        nodes=1
    shell:
        "python3 {input.script}"


rule all:
    input:
        rules.aggregate_results.output
