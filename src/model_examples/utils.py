import diffrax
import os
import pandas as pd
import jax.numpy as jnp
from typing import Callable, List, Tuple


def _load_csv_data(
        path: str,
        columns: List[str],
        time_col: str
) -> Tuple[jnp.ndarray, diffrax.SaveAt, List[float]]:
    df = pd.read_csv(path)
    data = jnp.array(df[columns].values)
    t_save = diffrax.SaveAt(ts=df[time_col].values)
    t_span = [0.0, float(df[time_col].max())]
    return data, t_save, t_span


def load_training_validation_csv(
    folder: str,
    train_file: str,
    validation_file: str,
    columns: List[str],
    time_col: str,
) -> Tuple[jnp.ndarray, diffrax.SaveAt, List[float], jnp.ndarray, diffrax.SaveAt, List[float]]:
    """Load training and validation data from CSV files."""
    base_path = os.path.join("src", "data_dir", folder)

    train_data, t_save_train, t_span_train = _load_csv_data(
        os.path.join(base_path, train_file), columns, time_col
    )
    val_data, t_save_val, t_span_val = _load_csv_data(
        os.path.join(base_path, validation_file), columns, time_col
    )

    return train_data, t_save_train, t_span_train, val_data, t_save_val, t_span_val


def get_result_plot_generic(
    prob,
    u0: jnp.ndarray,
    t_save: diffrax.SaveAt,
    t_span: List[float],
    data: jnp.ndarray,
    pred_transform: Callable[[jnp.ndarray], jnp.ndarray],
    data_transform: Callable[[jnp.ndarray], jnp.ndarray],
    labels: List[str],
):
    """Generate prediction and observation data frames for plotting."""
    sol = prob.solve_single_shooting(diffrax.Tsit5(), u0, t_save, t_span)
    pred_vals = pred_transform(sol.ys)
    obs_vals = data_transform(data)

    df_pred = pd.concat(
        [
            pd.DataFrame(
                {
                    "time": sol.ts,
                    "value": pred_vals[:, i],
                    "id": labels[i],
                    "datasource": "model",
                }
            )
            for i in range(pred_vals.shape[1])
        ]
    )

    df_obs = pd.concat(
        [
            pd.DataFrame(
                {
                    "time": sol.ts,
                    "value": obs_vals[:, i],
                    "id": labels[i],
                    "datasource": "observed",
                }
            )
            for i in range(obs_vals.shape[1])
        ]
    )

    return df_pred, df_obs
