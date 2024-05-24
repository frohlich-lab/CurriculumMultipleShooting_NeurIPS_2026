import diffrax
import equinox as eqx

from src.model_examples.lotka_volterra.setup_diffrax_model import UDEFunction as UDEFunction_lv
from src.model_examples.sir.setup_diffrax_model import UDEFunction as UDEFunction_sir
from src.ude_problem import CustomDropoutMLP
from typing import Any, Callable, Union


UDE_funcs = {
    'lotka_volterra': UDEFunction_lv,
    'sir': UDEFunction_sir,
}


class UDEProblem(eqx.Module):
    """
    Class to solve the UDEProblem with a chosen ODE-solver

    Methods
    -------
    solve(self, solver t_save, t_span)
        Solve the UDE for t_span = [t0, tend] and save the solution at the
        points in t_save using the provided solver. Controller is the adaptive
        ODE-stepping controller (should not need to be changed)
    """
    dynamical_model: str
    func: Union[UDEFunction_lv, UDEFunction_sir]

    def __init__(self, dynamical_model, dynamic_parameters, mlp, **kwargs):
        """
        Parameters
        ----------
        solver
            A valid diffrax ODE-solver (e.g. Tsit5 or Kvaerno5)
        dynamical_model: str
            The name of the dynamical model to use. Currently only 'lotka_volterra' and 'sir'.
        dynamic_parameters : jax.Array
            Starting values for the model dynamic parameters (α, δ in the model
            equations).
        u0 : jax.Array
            Initial values for the UDEProblem. Technically we can estimate these,
            but at least for the Lotka-Volterra model this is not the case.
        mlp : eqx.nn.MLP
            An equinox Standard Multi-Layer Perceptron (MLP) network. If set to
            None path_meta must be provided.
        controller : diffrax.PIDController
        """
        super().__init__(**kwargs)
        self.dynamical_model = dynamical_model
        if self.dynamical_model not in UDE_funcs.keys():
            raise ValueError(f"Unknown dynamical model: {dynamical_model}!")
        self.func = UDE_funcs[dynamical_model](dynamic_parameters, mlp)

    def solve(self, solver, u0, t_save, t_span,
              controller=diffrax.PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=1e-8, atol=1e-8),
              adjoint=diffrax.RecursiveCheckpointAdjoint()):
        t0, t1 = t_span
        sol = diffrax.diffeqsolve(diffrax.ODETerm(self.func),
                                  solver,
                                  t0,
                                  t1,
                                  None,
                                  u0,
                                  args=u0,
                                  saveat=t_save,
                                  throw=False,
                                  stepsize_controller=controller,
                                  adjoint=adjoint)
        return sol

    def copy(self, new_mlp=None):
        """
        Creates a copy of the UDEProblem with an optional new MLP for the UDEFunction.
        This allows to get a copy of the model for inference in case we are using Dropout layers in the MLP.

        Parameters
        ----------
        new_mlp : eqx.nn.MLP or eqx.Module, optional
            If provided, this will replace the existing MLP in the copied UDEFunction.

        Returns
        -------
        UDEProblem
            A new instance of UDEProblem with the specified modifications.
        """
        dynamical_model = self.dynamical_model
        dynamic_parameters = self.func.dynamic_parameters
        mlp = new_mlp if new_mlp is not None else self.func.mlp
        return UDEProblem(dynamical_model, dynamic_parameters, mlp)


def build_mlp(
        in_size: int,
        out_size: int,
        width_size: int,
        hidden_layers: int,
        key: Any,
        activation: Callable,
        use_dropout: bool,
        input_dropout_rate: float,
        hidden_dropout_rate: float,
) -> Union[eqx.nn.MLP, eqx.Module]:
    if not use_dropout:
        mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=hidden_layers,
            activation=activation,
            key=key
        )
    elif use_dropout and (input_dropout_rate is not None) and (hidden_dropout_rate is not None):
        mlp = CustomDropoutMLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=hidden_layers,
            activation=activation,
            key=key,
            dropout_rates={
                "input": input_dropout_rate,
                "hidden": hidden_dropout_rate,
            },
        )
    else:
        raise ValueError("For use_dropout to be True, dropout_rate cannot be None.")
    return mlp
