import diffrax
import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp

from jaxtyping import Array
from src.model_training.model_util import set_weight_bias
from src.ude_problem import build_mlp


class UDEFunctionLV(eqx.Module):
    dynamic_parameters: Array
    mlp: eqx.nn.MLP

    def __init__(self, dynamic_parameters, mlp, **kwargs):
        super().__init__(**kwargs)
        self.dynamic_parameters = dynamic_parameters
        self.mlp = mlp

    def __call__(self, t, u, args):
        alpha, delta = self.dynamic_parameters
        prey, predator = u
        nn_out = self.mlp(u)
        d_prey = alpha * prey + nn_out[0]
        d_predator = nn_out[1] - delta * predator
        return jnp.array([d_prey, d_predator])


class UDEProblemLV(eqx.Module):
    func: UDEFunctionLV

    def __init__(self, dynamic_parameters, mlp, **kwargs):
        super().__init__(**kwargs)
        self.func = UDEFunctionLV(dynamic_parameters, mlp)

    def solve(
            self, solver, u0, t_save, t_span,
            controller=diffrax.PIDController(
                pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=1e-12, atol=1e-12
            ),
            adjoint=diffrax.RecursiveCheckpointAdjoint()
        ):
        t0, t1 = t_span
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            solver,
            t0,
            t1,
            1e-6,
            u0,
            saveat=t_save,
            throw=False,
            stepsize_controller=controller,
            adjoint=adjoint,
            max_steps=20000
        )
        return sol


def setup_reference_problem(key_mlp, key_init):
    # Setup reference scenario
    true_u0 = jnp.array([0.44249296, 4.62805939])
    true_dynamic_parameters = jnp.array([1.3, 1.8])
    true_mlp = build_mlp(
        in_size=2,
        out_size=2,
        width_size=5,
        hidden_layers=2,
        key=key_mlp,
        use_bias=False,
        use_final_bias=False,
        activation=jnn.swish,
        final_activation=lambda x: x,
    )
    true_mlp = set_weight_bias(true_mlp, scale=0.1, key=key_init)


    true_ude_problem = UDEProblemLV(
        dynamic_parameters=true_dynamic_parameters,
        mlp=true_mlp
    )

    return true_u0, true_dynamic_parameters, true_ude_problem


def compute_reference_loss(prob, u0, t_save, t_span, data):
    u_pred = prob.solve(diffrax.Tsit5(), u0, t_save, t_span)
    return jnp.mean((u_pred.ys - data) ** 2)


def compute_loss_up_to_tmax(prob, u0, t_save, data, t_min, t_max):
    t_save_values = jnp.array([round(float(i), 3) for i in t_save.subs.ts])

    # Strict filtering within closed interval
    mask = (t_save_values >= t_min) & (t_save_values <= t_max)
    # print(jnp.sum(mask))
    t_save_in_range = t_save_values[mask]
    # print(t_save_in_range)

    cropped_t_save = diffrax.SaveAt(ts=t_save_in_range)
    cropped_data = data[mask, :]

    t_span = [t_min, t_max]
    return compute_reference_loss(prob, u0, cropped_t_save, t_span, cropped_data)