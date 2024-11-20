import diffrax
import equinox as eqx
import itertools as itt
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jaxtyping import Array
from src.common import plots_dir, Conf
from src.model_training.training_util import generate_incremental_arrays
from src.model_examples.lotka_volterra.setup_diffrax_model import compute_loss
from src.ude_problem import UDEProblem
from typing import Dict
from tqdm import tqdm


def analyse_lv_loss_landscape(
        udeprob: UDEProblem,
        u0: jnp.ndarray,
        train_data: jnp.ndarray,
        t_save_train: diffrax.SaveAt,
        t_span_train: list,
        validation_data: jnp.ndarray,
        t_save_validation: diffrax.SaveAt,
        t_span_validation: list,
        conf: Conf,
        N_EPOCHS: int,
        opt_dynamic_params: Dict[int, dict[str, Array]],
        num_points: int = 200,
):
    # We treat CL/no CL in the same way - no CL simply has a single fictitious CL step
    # Generate incremental training data for NN training - curriculum learning or not (if not, full data at each epoch)
    incremental_training_data, incremental_train_save, incremental_train_span, steps = generate_incremental_arrays(
        data=train_data,
        t_save=t_save_train,
        n_epoch=N_EPOCHS,
        curriculum_learning=conf.curriculum_learning,
        num_curriculum_steps=conf.num_curriculum_steps,
        max_curriculum_epochs=conf.max_curriculum_epochs
    )
    num_curriculum_steps = conf.num_curriculum_steps

    # Override steps and num_curriculum_steps if not using curriculum learning
    if not conf.curriculum_learning:
        steps = [0]  # fictitious step for non-curriculum learning
        num_curriculum_steps = 1

    # Set centre of the grid to the dynamic parameters of the found minimiser -- work in log-scale
    if conf.dynamic_params_scale == "log10":
        log_alpha, log_delta = udeprob.func.dynamic_parameters.tolist()
    else:
        log_alpha, log_delta = np.log10(udeprob.func.dynamic_parameters).tolist()

    # Define scan ranges for alpha and delta spanning sampling range
    alpha_scan_range, delta_scan_range = [
        np.linspace(
            log_param - conf.multistart_deviation,
            log_param + conf.multistart_deviation,
            num=num_points
        )
        for log_param in [log_alpha, log_delta]
    ]

    dfs = []

    for alpha_test, delta_test in tqdm(itt.product(alpha_scan_range, delta_scan_range)):
        # Replace dynamic parameters and partition
        new_udeprob = udeprob.copy(
            new_dynamic_parameters=jnp.array([alpha_test, delta_test]),
            new_options={"dynamic_params_scale": "log10"}  # ensure parameters are on log10 scale since we are initialising in log scale
        )
        params, static = eqx.partition(new_udeprob, eqx.is_array)

        # compute validation loss one time only
        loss_val, _ = compute_loss(
            prob_params=params, prob_static=static,
            u0=u0,
            t_save=t_save_validation,
            t_span=t_span_validation,
            data=validation_data,
            mlp_reg_exp=conf.mlp_reg_exp, mlp_reg_strength=conf.mlp_reg_strength,
            output_reg=conf.output_reg,
        )
        # compute training loss once per curriculum stage
        for curriculum_step in range(num_curriculum_steps):
            loss_train, _ = compute_loss(
                prob_params=params, prob_static=static,
                u0=u0,
                t_save=incremental_train_save[steps[curriculum_step]],
                t_span=incremental_train_span[steps[curriculum_step]],
                data=incremental_training_data[steps[curriculum_step]],
                mlp_reg_exp=conf.mlp_reg_exp, mlp_reg_strength=conf.mlp_reg_strength,
                output_reg=conf.output_reg,
            )
            dfs.append(
                pd.DataFrame(
                    {
                        "alpha": [alpha_test],
                        "delta": [delta_test],
                        "loss_train": [loss_train],
                        "loss_val": [loss_val],
                        "curriculum_step": [curriculum_step]
                    }
                )
            )

    # Concatenate resulting dataframes and convert loss values to floats
    df = pd.concat(dfs)
    df["loss_train"] = df["loss_train"].astype(float)
    df["loss_val"] = df["loss_val"].astype(float)

    # ================================================================
    #                            PLOTTING
    # ================================================================
    alpha_values = df['alpha'].unique()
    delta_values = df['delta'].unique()
    alpha_mesh, delta_mesh = np.meshgrid(alpha_values, delta_values)

    plt.subplots(1, conf.num_curriculum_steps, figsize=(10 * conf.num_curriculum_steps, 6))
    for curriculum_step in df.curriculum_step.unique():
        # Reshaping loss_train values to fit the meshgrid
        loss_mesh = np.log10(
            df[df.curriculum_step==curriculum_step].pivot(
                index='delta', columns='alpha', values=f'loss_train'
            ).values
        )

        # Creating the contour plot
        plt.subplot(1, conf.num_curriculum_steps, curriculum_step + 1)
        # Draw contour lines
        plt.contour(alpha_mesh, delta_mesh, loss_mesh, levels=50, alpha=0.1)
        # Fill contour
        contour = plt.contourf(alpha_mesh, delta_mesh, loss_mesh, levels=50, cmap='coolwarm')
        # Add dynamic params for start/end of CL step in black/magenta stars, respectively
        for timepoint, color in zip(["start", "end"], ["black", "magenta"]):
            plt.scatter(
                opt_dynamic_params[curriculum_step][timepoint][0],
                opt_dynamic_params[curriculum_step][timepoint][1],
                marker='*',
                c=color,
                s=300
            )
        # Add reference lines
        plt.axvline(log_alpha, c='gray', ls='--')
        plt.axhline(log_delta, c='gray', ls='--')
        plt.colorbar(contour)
        plt.xlabel('Log10 Alpha')
        plt.ylabel('Log10 Delta')
        plt.title(f'Loss train - {curriculum_step}')
    plt.savefig(plots_dir / f"two_stage_loss_train_CL.pdf")
    plt.savefig(plots_dir / f"two_stage_loss_train_CL.svg")
    plt.show()


    return df
