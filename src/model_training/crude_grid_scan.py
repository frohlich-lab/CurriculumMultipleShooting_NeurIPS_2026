import itertools as itt
import jax.random as jr
import jax.nn as jnn
import jax.numpy as jnp
import src.model_examples.lotka_volterra.setup_diffrax_model as setup_lv
import src.model_examples.sir.setup_diffrax_model as setup_sir

from src.common import build_mlp, Conf, EarlyStoppingParams, set_weight_bias
from src.model_training.training import train
from src.model_training.training_configuration import (DYNAMICAL_MODEL,
                                                       MLP_WIDTHS,
                                                       MLP_DEPTHS,
                                                       MLP_ACTIVATIONS,
                                                       OPTIMISERS,
                                                       USE_EARLY_STOP,
                                                       MAX_LEARNING_RATES,
                                                       LEARNING_RATE_SPANS,
                                                       LEARNING_RATE_DECAYS,
                                                       WARMUP_FCTS,
                                                       OPT_STEPS,
                                                       OPT_MULT,
                                                       REG_EXPS,
                                                       REG_STRENGTHS,
                                                       N_EPOCHS,
                                                       MAX_ITER_OPTX)

# Dictionary of activation functions to choose from
activation_functions = {
    "tanh": jnn.tanh,
    "relu": jnn.relu,
    "swish": jnn.swish,
    "elu": jnn.elu,
    "gelu": jnn.gelu,
}

# Key
key = jr.PRNGKey(123)
key_mlp, key_init = jr.split(key, 2)

# Setup default u0, dynamic parameters, in_size, out_size
u0 = {
    'lotka_volterra': jnp.array([0.44249296, 4.6280594]),
    'sir': jnp.array([51000000.0, 578, 24, 10]),
}

dynamic_parameters = {
    'lotka_volterra': jnp.array([1.3, 1.8]),
    'sir': jnp.array([0.15, 0.013, 0.01]),
}

in_size = {
    'lotka_volterra': 2,
    'sir': 3,
}

out_size = {
    'lotka_volterra': 2,
    'sir': 1,
}

for (
        dynamical_model,
        mlp_layer_width, mlp_depth, activation_fn_name,
        optimiser, max_lrate, lrate_span, lrate_decay, warmup_fct, opt_steps, opt_mult,
        use_early_stop, reg_exp, reg_strength
) in itt.product(
    DYNAMICAL_MODEL,
    MLP_WIDTHS, MLP_DEPTHS, MLP_ACTIVATIONS,
    OPTIMISERS, MAX_LEARNING_RATES, LEARNING_RATE_SPANS, LEARNING_RATE_DECAYS, WARMUP_FCTS, OPT_STEPS, OPT_MULT,
    USE_EARLY_STOP, REG_EXPS, REG_STRENGTHS,
):
    conf = Conf(
        dynamical_model=dynamical_model,
        mlp_layer_width=mlp_layer_width,
        mlp_depth=mlp_depth,
        activation_fn_name=activation_fn_name,
        optimiser=optimiser,
        max_lrate=max_lrate,
        lrate_span=lrate_span,
        lrate_decay=lrate_decay,
        warmup_fct=warmup_fct,
        opt_steps=opt_steps,
        opt_mult=opt_mult,
        use_early_stop=use_early_stop,
        reg_exp=reg_exp,
        reg_strength=reg_strength,
    )

    # Setup MLP (same as the original paper for now)
    mlp = build_mlp(
        in_size=in_size[conf.dynamical_model],
        out_size=out_size[conf.dynamical_model],
        width_size=conf.mlp_layer_width,
        hidden_layers=conf.mlp_depth,
        key=key_mlp,
        activation=activation_functions[conf.activation_fn_name],
    )
    # Initialise weights and biases of MLP
    mlp = set_weight_bias(mlp, conf.init_scale, key_init)

    # Setup UDEProblem to train
    if conf.dynamical_model == 'lotka_volterra':
        # TODO @GiacomoFabrini find better way to express filepath and pass it as argument
        (train_data, t_save_train, t_span_train,
         validation_data, t_save_validation, t_span_validation) = setup_lv.get_training_validation_data()

        ude_problem = setup_lv.UDEProblem(dynamic_parameters[conf.dynamical_model], mlp)
        loss_fn = setup_lv.compute_loss
        loss_grad_fn = setup_lv.compute_loss_grad
        loss_optx = setup_lv.compute_loss_optx
        optx_config = {
            "rtol": 1e-3,
            "atol": 1e-3,
        }
    elif conf.dynamical_model == 'sir':
        # TODO @GiacomoFabrini find better way to express filepath and pass it as argument - same as above
        (train_data, t_save_train, t_span_train,
         validation_data, t_save_validation, t_span_validation) = setup_sir.get_training_validation_data()

        ude_problem = setup_sir.UDEProblem(dynamic_parameters[conf.dynamical_model], mlp)
        loss_fn = setup_sir.compute_loss
        loss_grad_fn = setup_sir.compute_loss_grad
        loss_optx = setup_sir.compute_loss_optx
        optx_config = {
            "rtol": 1e-4,
            "atol": 1e-4,
        }
    else:
        raise ValueError(f"Unknown dynamical model: {conf.dynamical_model}!")

    # Linear schedule Fabian set up for the DMM
    # schedule_config = dict(
    #     init_value=conf.optimiser_learning_rate,
    #     transition_steps=N_EPOCHS/10,  # 100 when using 1000 epochs
    #     end_value=conf.optimiser_learning_rate/10
    # )

    early_stopping_params = EarlyStoppingParams(
        use_early_stopping=conf.use_early_stop,  # enables flax.training.early_stopping
        patience=50,
        # number of consecutive epochs where we tolerate loss_val not improving by at least min_improvement
        min_improvement=0,  # absolute improvement in loss_val to consider as improvement
    )

    trained_model = train(
        model=ude_problem,
        u0=u0[conf.dynamical_model],
        training_data=train_data,
        t_save_train=t_save_train,
        t_span_train=t_span_train,
        validation_data=validation_data,
        t_save_val=t_save_validation,
        t_span_val=t_span_validation,
        conf=conf,
        n_epochs=N_EPOCHS,
        early_stopping_params=early_stopping_params,
        loss_fn=loss_fn,
        loss_grad_fn=loss_grad_fn,
        loss_optx=loss_optx,
        optx_config=optx_config,
        max_iter_optx=MAX_ITER_OPTX,
    )
