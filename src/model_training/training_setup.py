import jax.numpy as jnp

from src.common import activation_functions, Conf, default_out_sizes, u0s
from src.model_examples.lotka_volterra import setup_diffrax_model as setup_lv
from src.model_examples.sir import setup_diffrax_model as setup_sir
from src.model_training.model_util import set_weight_bias
from src.ude_problem import build_mlp


def get_io_size(conf, u0):
    if conf.dynamical_model == 'lotka_volterra':
        in_size = len(u0)
    elif conf.dynamical_model == 'sir':
        in_size = len(u0) - 1
    else:
        raise ValueError(f"Unknown dynamical model: {conf.dynamical_model}!")
    out_size = default_out_sizes[conf.dynamical_model] + conf.node_augmentation
    return in_size, out_size


def get_data_and_funcs_for_train(conf: Conf, dynamic_params_shape: tuple) -> tuple:
    if conf.dynamical_model == 'lotka_volterra':
        # TODO @GiacomoFabrini find better way to express filepath and pass it as argument
        (train_data, t_save_train, t_span_train,
         validation_data, t_save_validation, t_span_validation) = setup_lv.get_training_validation_data()

        loss_fn = setup_lv.compute_loss
        loss_grad_fn = setup_lv.compute_loss_grad
        loss_optx = setup_lv.compute_loss_optx
        optx_config = {
            "rtol": 1e-6,  # changed from 1e-3
            "atol": 1e-6,  # changed from 1e-3
        }
    elif conf.dynamical_model == 'sir':
        # TODO @GiacomoFabrini find better way to express filepath and pass it as argument - same as above
        (train_data, t_save_train, t_span_train,
         validation_data, t_save_validation, t_span_validation) = setup_sir.get_training_validation_data()

        loss_fn = setup_sir.compute_loss
        loss_grad_fn = setup_sir.compute_loss_grad
        loss_optx = setup_sir.compute_loss_optx
        optx_config = {
            "rtol": 1e-6,  # changed from 1e-4
            "atol": 1e-6,  # changed from 1e-4
        }
    else:
        raise ValueError(f"Unknown dynamical model: {conf.dynamical_model}!")
    options = {
        "dynamic_params_scale": conf.dynamic_params_scale,
        "bounded_tanh_params": broadcast_bounds(
            bounds={"lb": conf.dynamic_params_lb, "ub": conf.dynamic_params_ub},
            shape=dynamic_params_shape
        )  # bounds are ignored if not using bounded tanh transform from Philipps et al. 2024
    }
    train_args = train_data, t_save_train, t_span_train
    val_args = validation_data, t_save_validation, t_span_validation
    loss_args = loss_fn, loss_grad_fn, loss_optx
    option_args = optx_config, options
    return train_args, val_args, loss_args, option_args


def broadcast_bounds(bounds, shape):
    return {
        key: jnp.broadcast_to(value, shape)
        for key, value in bounds.items()
    }


def get_initial_conditions(conf: Conf):
    u0 = u0s[conf.dynamical_model]
    # Apply NODE-like augmentation
    if conf.node_augmentation > 0:
            u0 = jnp.concatenate([u0, jnp.zeros(conf.node_augmentation)])
    # Ensure 64-bit precision
    u0 = u0.astype(jnp.float64)
    return u0


def setup_mlp(conf: Conf, u0: jnp.ndarray, key_mlp, key_init):
    # Define input and output sizes for MLP
    in_size, out_size = get_io_size(conf, u0)

    # Check activation function is valid
    if conf.activation_fn_name not in activation_functions:
        raise ValueError(f"Unknown activation function: {conf.activation_fn_name}!")

    # Build the MLP, initialise its weights and biases
    mlp = set_weight_bias(
        build_mlp(
            in_size=in_size,
            out_size=out_size,
            width_size=conf.mlp_layer_width,
            hidden_layers=conf.mlp_depth,
            key=key_mlp,
            use_bias=conf.use_bias,
            use_final_bias=conf.use_final_bias,
            activation=activation_functions[conf.activation_fn_name],
            final_activation=activation_functions[conf.final_activation_fn_name],
            use_dropout=conf.use_dropout,
            input_dropout_rate=conf.input_dropout_rate,
            hidden_dropout_rate=conf.hidden_dropout_rate,
            use_skip_connections=conf.use_skip_connections,
        ),
        conf.init_scale,
        key_init
    )
    return mlp
