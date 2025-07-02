import diffrax
import equinox as eqx
import jax
import jax.random as jr
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.common import Conf, state_dims
from src.model_examples.lotka_volterra import setup_lotka_volterra_model as setup_lv
from src.model_training.base_losses import get_mse
from src.model_training.training_setup import (
    get_data_and_funcs_for_train,
    get_initial_conditions,
    setup_mlp,
    setup_time_masks_u0_curriculum
)
from src.model_training.training_util import get_sol_and_loss_single_shooting, get_sol_and_loss_validation
from src.ude_problem import UDEProblem
from tests.test_loss_reference import setup_reference_problem, compute_reference_loss, compute_loss_up_to_tmax

# Set jax precision
jax.config.update("jax_enable_x64", True)

# Set random seed and get random keys
seed = 0
key = jr.PRNGKey(seed)
key_mlp, key_init = jr.split(key, 2)

# Setup reference problem
true_u0, true_dynamic_parameters, true_ude_problem = setup_reference_problem(key_mlp, key_init)

(
    train_data, t_save_train, t_span_train,
    validation_data, t_save_validation, t_span_validation
) = setup_lv.get_training_validation_data()


t_maxes = {
    3: [1.90, 3.90, 6.10],
    5: [1.10, 2.30, 3.50, 4.70, 6.10],
    6: [0.90, 1.90, 2.90, 3.90, 4.90, 6.10],
    10: [0.50, 1.10, 1.70, 2.30, 2.90, 3.50, 4.10, 4.70, 5.30, 6.10],
}

num_curriculum_steps = 6

true_losses = []
for cl_stage, t_max in enumerate(t_maxes[num_curriculum_steps]):
    loss_train = compute_loss_up_to_tmax(
        true_ude_problem,
        true_u0,
        t_save_train,
        train_data,
        0,  # all curriculum stages start from 0
        t_max
    )

    print(f"CL stage {cl_stage+1}, True Training Loss: {loss_train}")
    true_losses.append(loss_train)
true_losses = jnp.array(true_losses)

# Set up our current scenario to test
conf = Conf(
    dynamical_model="lotka_volterra",
    mlp_layer_width=5,
    mlp_depth=2,
    activation_fn_name="swish",
    final_activation_fn_name="identity",
    use_bias=False,
    use_final_bias=False,
    shooting_mode="single",
    curriculum_learning=True,
    num_curriculum_steps=num_curriculum_steps,
    multistart_deviation=0,
    multistart_job=seed,
    dynamic_params_scale="linear",
)

# Get initial conditions
u0 = get_initial_conditions(conf)

# ================================================================
#                         SET UP MLP
# ================================================================
mlp = setup_mlp(conf, u0, key_mlp, key_init)

assert jnp.allclose(true_ude_problem.func.mlp(true_u0), mlp(u0), rtol=1e-10, atol=1e-10), \
    "MLP setup does not match the true MLP used for testing!"

# ================================================================
#                   SET UP UDE PROBLEM TO TRAIN
# ================================================================
# Get data and functions for training
(
    train_args, val_args,
    loss_dict, option_args, plot_args, rescaling_args
) = get_data_and_funcs_for_train(
    conf, true_dynamic_parameters.shape, u0
)
optx_config, options = option_args

# Set up SaveAt windows, binary masks and initial values
t_save_windows, masks, initial_values, curriculum_stage_dict = setup_time_masks_u0_curriculum(
    conf=conf,
    train_args=train_args,
    u0_guess=train_args.u0,
    effective_state_dimension=state_dims[conf.dynamical_model] + conf.node_augmentation,
)
# Update plot args
plot_args["t_save_windows"] = t_save_windows

ude_problem = UDEProblem(
    dynamical_model=conf.dynamical_model,
    dynamic_parameters=true_dynamic_parameters,
    mlp=mlp,
    options=options,
    state_dimension=state_dims[conf.dynamical_model],
    initial_values=initial_values,
    regularise_network=False,
)

params, static = eqx.partition(ude_problem, eqx.is_array)
our_losses = []
for cl_stage in range(1, conf.num_curriculum_steps+1):
    t_save_windows = curriculum_stage_dict[cl_stage]["time"]
    masks = curriculum_stage_dict[cl_stage]["mask"]

    our_loss, _ = loss_dict["loss_fn"](
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
    print(f"{cl_stage}, our training loss = {our_loss}")
    our_losses.append(our_loss)

our_losses = jnp.array(our_losses)

plt.plot(jnp.abs(true_losses - our_losses))
plt.xticks(range(num_curriculum_steps))
plt.xlabel("Curriculum Stage")
plt.ylabel("Absolute loss deviation from reference")
plt.show()
print("Done")