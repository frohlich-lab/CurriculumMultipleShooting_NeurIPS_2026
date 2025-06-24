import diffrax
import os
import pandas as pd
import jax.numpy as jnp
from typing import List, Tuple


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
