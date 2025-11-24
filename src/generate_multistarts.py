import pandas as pd
import numpy as np
from scipy.stats import qmc

from src.common import data_dir, true_dynamic_parameters
from src.generate_run_configs import generate_param_combinations
from src.model_training.training_configuration import N_STARTS


def get_unique_sampling_settings(param_combinations):
    settings = set()
    for p in param_combinations:
        settings.add((
            p["dynamical_model"],
            p["multistart_deviation"],
            p["dynamic_params_perturb_method"],
        ))
    return list(settings)


def generate_multistart_params(
    true_params: np.ndarray, n_starts: int, method: str, deviation: float, seed: int = 0
) -> np.ndarray:
    dim = true_params.shape[0]
    log_true_params = np.log10(true_params)

    if method == "lh":
        sampler = qmc.LatinHypercube(d=dim, seed=seed)
        sample = sampler.random(n=n_starts)
        sample = 2 * (sample - 0.5)  # [-1, 1)
        log_params = log_true_params[None, :] + deviation * sample
    elif method == "su":
        low = log_true_params - deviation
        high = log_true_params + deviation
        sample = np.random.default_rng(seed).uniform(low, high, size=(n_starts, dim))
        log_params = sample
    elif method == "agn":
        sample = np.random.default_rng(seed).normal(loc=0.0, scale=deviation, size=(n_starts, dim))
        log_params = log_true_params[None, :] + sample
    else:
        raise ValueError(f"Unknown method: {method}")

    return log_params


def main():
    param_combinations = generate_param_combinations(N_STARTS)
    sampling_settings = get_unique_sampling_settings(param_combinations)

    for model, deviation, method in sampling_settings:
        true_params = np.array(true_dynamic_parameters[model])
        sampled_log_params = generate_multistart_params(
            true_params=true_params,
            n_starts=N_STARTS,
            method=method,
            deviation=deviation,
        )

        df = pd.DataFrame(
            sampled_log_params,
            columns=[f"param_{i}" for i in range(sampled_log_params.shape[1])]
        )
        df.insert(loc=0, column="multistart_job", value=np.array(list(range(N_STARTS))))

        out_path = data_dir / model / f"{method}__{deviation}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
