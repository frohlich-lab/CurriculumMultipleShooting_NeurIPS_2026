import pytest

import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import os
import sys

from pathlib import Path
from typing import Optional

current_dir = Path(os.getcwd())
sys.path.append(str(current_dir))

from src.common import Conf, state_dims
from src.model_examples.lotka_volterra import setup_lotka_volterra_model as setup_lv
from src.model_training.training_setup import (
    get_data_and_funcs_for_train,
    get_initial_conditions,
    setup_mlp,
    setup_time_masks_u0_curriculum,
)
from src.ude_problem import UDEProblem
from tests.test_loss_reference import (
    setup_reference_problem,
    compute_reference_loss,
    compute_loss_up_to_tmax,
)

# Reference windows and curriculum settings used in the original tests
T_MINS = {
    6: [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
    8: [0, 0.7, 1.4, 2.1, 2.8, 3.5, 4.2, 4.9, 5.6],
    14: [0, 1.3, 2.6, 3.9, 5.2],
}

T_MAXES = {
    6: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.1],
    8: [0.7, 1.4, 2.1, 2.8, 3.5, 4.2, 4.9, 5.6, 6.1],
    14: [1.3, 2.6, 3.9, 5.2, 6.1],
}

CURRICULUM_TMAXES = {
    3: [1.90, 4.00, 6.10],
    5: [1.10, 2.30, 3.60, 4.80, 6.10],
    6: [0.90, 1.90, 3.00, 4.00, 5.00, 6.10],
    9: [0.50, 1.20, 1.90, 2.60, 3.30, 4.00, 4.70, 5.40, 6.10],
    10: [0.50, 1.10, 1.70, 2.30, 3.00, 3.60, 4.20, 4.80, 5.40, 6.10],
    13: [0.30, 0.80, 1.30, 1.80, 2.20, 2.70, 3.20, 3.70, 4.10, 4.60, 5.10, 5.60, 6.10],
}


@pytest.fixture(scope="module")
def reference_setup():
    jax.config.update("jax_enable_x64", True)
    seed = 0
    key = jr.PRNGKey(seed)
    key_mlp, key_init = jr.split(key, 2)
    true_u0, true_dynamic_parameters, true_ude_problem = setup_reference_problem(key_mlp, key_init)
    (
        train_data, t_save_train, t_span_train,
        validation_data, t_save_validation, t_span_validation,
    ) = setup_lv.get_training_validation_data()
    return {
        "seed": seed,
        "key_mlp": key_mlp,
        "key_init": key_init,
        "true_u0": true_u0,
        "true_dynamic_parameters": true_dynamic_parameters,
        "true_ude_problem": true_ude_problem,
        "train_data": train_data,
        "t_save_train": t_save_train,
        "t_span_train": t_span_train,
        "validation_data": validation_data,
        "t_save_validation": t_save_validation,
        "t_span_validation": t_span_validation,
    }


def build_conf(
    shooting_mode: str,
    curriculum: bool,
    seed: int,
    *,
    num_curriculum_steps: Optional[int] = None,
    shooting_window_length: Optional[int] = None,
) -> Conf:
    kwargs = dict(
        dynamical_model="lotka_volterra",
        mlp_layer_width=5,
        mlp_depth=2,
        activation_fn_name="swish",
        final_activation_fn_name="identity",
        use_bias=False,
        use_final_bias=False,
        shooting_mode=shooting_mode,
        curriculum_learning=curriculum,
        multistart_deviation=0,
        multistart_job=seed,
        dynamic_params_scale="linear",
    )
    if shooting_mode == "multiple":
        kwargs.update(
            shooting_window_length=shooting_window_length or 8,
            shooting_iv_init="g",
            output_reg=0,
            multiple_shooting_penalty=0,
        )
    if curriculum and shooting_mode == "single":
        kwargs["num_curriculum_steps"] = num_curriculum_steps or 5
    return Conf(**kwargs)


def compute_true_losses(ref, conf: Conf):
    prob = ref["true_ude_problem"]
    u0 = ref["true_u0"]
    t_save_train = ref["t_save_train"]
    train_data = ref["train_data"]

    t_mins = T_MINS
    t_maxes = T_MAXES
    curriculum_tmaxes = CURRICULUM_TMAXES

    if conf.shooting_mode == "single" and not conf.curriculum_learning:
        return compute_reference_loss(prob, u0, t_save_train, ref["t_span_train"], train_data)

    if conf.shooting_mode == "single" and conf.curriculum_learning:
        values = curriculum_tmaxes[conf.num_curriculum_steps]
        losses = [
            compute_loss_up_to_tmax(prob, u0, t_save_train, train_data, 0, t_max)
            for t_max in values
        ]
        return jnp.array(losses)

    if conf.shooting_mode == "multiple" and not conf.curriculum_learning:
        length = conf.shooting_window_length
        losses = [
            compute_loss_up_to_tmax(prob, u0, t_save_train, train_data, t_min, t_max)
            for t_min, t_max in zip(t_mins[length], t_maxes[length])
        ]
        return jnp.mean(jnp.array(losses))

    # multiple shooting with curriculum
    length = conf.shooting_window_length
    num_windows = int(jnp.ceil(len(train_data) / (length - 1)))
    cl_losses = []
    for cl_stage in range(1, num_windows + 1):
        window_losses = []
        cl_t_mins = t_mins[length][: -(cl_stage - 1)] if cl_stage > 1 else t_mins[length]
        cl_t_maxes = t_maxes[length][(cl_stage - 1) :] if cl_stage > 1 else t_maxes[length]
        for t_min, t_max in zip(cl_t_mins, cl_t_maxes):
            window_losses.append(
                compute_loss_up_to_tmax(prob, u0, t_save_train, train_data, t_min, t_max)
            )
        cl_losses.append(jnp.mean(jnp.array(window_losses)))
    return jnp.array(cl_losses)


def compute_our_losses(ref, conf: Conf) -> jnp.ndarray:
    key_mlp, key_init = ref["key_mlp"], ref["key_init"]
    u0_guess = get_initial_conditions(conf)
    mlp = setup_mlp(conf, u0_guess, key_mlp, key_init)
    assert jnp.allclose(ref["true_ude_problem"].func.mlp(ref["true_u0"]), mlp(u0_guess), rtol=1e-10, atol=1e-10)

    (
        train_args,
        _val_args,
        loss_dict,
        option_args,
        _plot_args,
        _rescaling_args,
    ) = get_data_and_funcs_for_train(conf, ref["true_dynamic_parameters"].shape, u0_guess)
    _optx_config, options = option_args

    t_save_windows, masks, initial_values, curriculum_stage_dict = setup_time_masks_u0_curriculum(
        conf=conf,
        train_args=train_args,
        u0_guess=train_args.u0,
        effective_state_dimension=state_dims[conf.dynamical_model] + conf.node_augmentation,
    )

    ude_problem = UDEProblem(
        dynamical_model=conf.dynamical_model,
        dynamic_parameters=ref["true_dynamic_parameters"],
        mlp=mlp,
        options=options,
        state_dimension=state_dims[conf.dynamical_model],
        initial_values=initial_values,
        regularise_network=conf.shooting_mode == "multiple",
    )

    params, static = eqx.partition(ude_problem, eqx.is_array)

    if conf.shooting_mode == "single" and not conf.curriculum_learning:
        loss, _ = loss_dict["loss_fn"](
            prob_params=params,
            prob_static=static,
            t_save_windows=t_save_windows,
            masks=masks,
            data=train_args.data,
            last_valid_index=1,
            mlp_reg_exp=conf.mlp_reg_exp,
            mlp_reg_strength=0,
            output_reg=0,
            multiple_shooting_penalty=0,
            shooting_mode=conf.shooting_mode,
        )
        return loss

    if conf.shooting_mode == "single" and conf.curriculum_learning:
        losses = []
        for stage in range(1, conf.num_curriculum_steps + 1):
            stage_loss, _ = loss_dict["loss_fn"](
                prob_params=params,
                prob_static=static,
                t_save_windows=curriculum_stage_dict[stage]["time"],
                masks=curriculum_stage_dict[stage]["mask"],
                data=train_args.data,
                last_valid_index=1,
                mlp_reg_exp=conf.mlp_reg_exp,
                mlp_reg_strength=0,
                output_reg=0,
                multiple_shooting_penalty=0,
                shooting_mode=conf.shooting_mode,
            )
            losses.append(stage_loss)
        return jnp.array(losses)

    num_windows = int(jnp.ceil(len(train_args.data) / (conf.shooting_window_length - 1)))
    if not conf.curriculum_learning:
        loss, _ = loss_dict["loss_fn"](
            prob_params=params,
            prob_static=static,
            t_save_windows=t_save_windows,
            masks=masks,
            data=train_args.data,
            last_valid_index=num_windows,
            mlp_reg_exp=conf.mlp_reg_exp,
            mlp_reg_strength=0,
            output_reg=0,
            multiple_shooting_penalty=0,
            shooting_mode=conf.shooting_mode,
        )
        return loss

    losses = []
    for stage in range(1, num_windows + 1):
        stage_loss, _ = loss_dict["loss_fn"](
            prob_params=params,
            prob_static=static,
            t_save_windows=curriculum_stage_dict[stage]["time"],
            masks=curriculum_stage_dict[stage]["mask"],
            data=train_args.data,
            last_valid_index=num_windows - (stage - 1),
            mlp_reg_exp=conf.mlp_reg_exp,
            mlp_reg_strength=0,
            output_reg=0,
            multiple_shooting_penalty=0,
            shooting_mode=conf.shooting_mode,
        )
        losses.append(stage_loss)
    return jnp.array(losses)


PARAMS = [
    ("single", False, None),
    *(("single", True, n) for n in CURRICULUM_TMAXES.keys()),
    *(("multiple", False, l) for l in T_MINS.keys()),
    *(("multiple", True, l) for l in T_MINS.keys()),
]

IDS = ["single-no-curriculum"]
IDS += [f"single-curriculum-{n}" for n in CURRICULUM_TMAXES.keys()]
IDS += [f"multiple-{l}" for l in T_MINS.keys()]
IDS += [f"multiple-curriculum-{l}" for l in T_MINS.keys()]


@pytest.mark.parametrize(
    "shooting_mode,curriculum,param",
    PARAMS,
    ids=IDS,
)
def test_training_scheme_loss(shooting_mode, curriculum, param, reference_setup):
    kwargs = {}
    if shooting_mode == "single" and curriculum:
        kwargs["num_curriculum_steps"] = param
    if shooting_mode == "multiple":
        kwargs["shooting_window_length"] = param
    conf = build_conf(shooting_mode, curriculum, reference_setup["seed"], **kwargs)
    true_losses = compute_true_losses(reference_setup, conf)
    our_losses = compute_our_losses(reference_setup, conf)
    assert jnp.allclose(our_losses, true_losses, rtol=1e-4, atol=1e-4)
