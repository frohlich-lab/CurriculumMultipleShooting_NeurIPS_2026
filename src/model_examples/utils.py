import diffrax
import os
import pandas as pd
import jax.numpy as jnp
from typing import Callable, List, Tuple


def load_training_validation_csv(
    folder: str,
    train_file: str,
    validation_file: str,
    columns: List[str],
    time_col: str,
) -> Tuple[jnp.ndarray, diffrax.SaveAt, List[float], jnp.ndarray, diffrax.SaveAt, List[float]]:
    """Load training and validation data from CSV files.

    Parameters
    ----------
    folder : str
        Folder under ``src/data_dir`` containing the CSV files.
    train_file : str
        Name of the training CSV file.
    validation_file : str
        Name of the validation CSV file.
    columns : List[str]
        Columns to extract from the CSV files as data.
    time_col : str
        Column name containing the time points.
    """
    train_df = pd.read_csv(os.path.join("src", "data_dir", folder, train_file))
    train_data = jnp.array(train_df[columns].values)
    t_save_train = diffrax.SaveAt(ts=train_df[time_col].values)
    t_span_train = [0.0, float(train_df[time_col].max())]

    val_df = pd.read_csv(os.path.join("src", "data_dir", folder, validation_file))
    val_data = jnp.array(val_df[columns].values)
    t_save_val = diffrax.SaveAt(ts=val_df[time_col].values)
    t_span_val = [0.0, float(val_df[time_col].max())]

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
        ],
        names=False,
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
        ],
        names=False,
    )

    return df_pred, df_obs
