import os
import pandas as pd

from generate_run_configs import generate_param_combinations
from src.common import TRAINING_RESULTS, MERGED_TRAINING_RESULTS
from src.model_training.training_configuration import DYNAMICAL_MODEL, EXPERIMENT_NAME, N_STARTS


dfs = {dynamical_model: [] for dynamical_model in DYNAMICAL_MODEL}

for hyperparam_config in generate_param_combinations(N_STARTS):
    training_result_file = TRAINING_RESULTS.format(**hyperparam_config, experiment_name=EXPERIMENT_NAME)
    if os.path.exists(training_result_file):
        df = pd.read_csv(training_result_file)
        dfs[hyperparam_config["dynamical_model"]].append(df)

for dynamical_model in DYNAMICAL_MODEL:
    if len(dfs[dynamical_model]) == 0:
        print(f"No results for {dynamical_model}")
        continue
    else:
        final_df = pd.concat(dfs[dynamical_model])
        final_df.to_csv(
            str(MERGED_TRAINING_RESULTS).format(
                dynamical_model=dynamical_model,
                experiment_name=EXPERIMENT_NAME
            )
        )
