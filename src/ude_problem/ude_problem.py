import diffrax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import json
import numpy as np

from pathlib import Path

from src.common import activation_functions
from src.model_examples.lotka_volterra.setup_diffrax_model import UDEFunction as UDEFunction_lv
from src.model_examples.sir.setup_diffrax_model import UDEFunction as UDEFunction_sir
from src.ude_problem import CustomDropoutMLP
from typing import Any, Callable, Union
from src.ude_problem import CustomDropoutMLP, ResidualMLP


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

    def __init__(
            self, dynamical_model, dynamic_parameters, mlp, options, state_dimension, regularise_network=None, **kwargs
    ):
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
        options: dict
            Options for the UDEFunction.
        controller : diffrax.PIDController
        """
        super().__init__(**kwargs)
        self.dynamical_model = dynamical_model
        if self.dynamical_model not in UDE_funcs.keys():
            raise ValueError(f"Unknown dynamical model: {dynamical_model}!")
        self.func = UDE_funcs[dynamical_model](dynamic_parameters, mlp, options, state_dimension, regularise_network)

    def solve(self, solver, u0, t_save, t_span,
              controller=diffrax.PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=1e-8, atol=1e-8),
              adjoint=diffrax.RecursiveCheckpointAdjoint()):
        if not isinstance(t_save, diffrax.SaveAt):
            t_save = diffrax.SaveAt(ts=t_save)
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

    def copy(
            self,
            new_mlp: Union[eqx.nn.MLP, CustomDropoutMLP] = None,
            new_regularise_network: bool = None,
            new_dynamic_parameters: jnp.ndarray = None,
            new_options: dict = None,
    ):
        """
        Creates a copy of the UDEProblem with an optional new MLP for the UDEFunction.
        This allows to get a copy of the model for inference in case we are using Dropout layers in the MLP.

        Parameters
        ----------
        new_mlp : eqx.nn.MLP or eqx.Module, optional
            If provided, this will replace the existing MLP in the copied UDEFunction.
        new_regularise_network: bool
            If provided, this will replace the existing regularise_network flag in the copied UDEFunction.
        new_dynamic_parameters: jnp.ndarray
            If provided, this will overwrite the existing self.func.dynamic_parameters
        new_options: dict
            If provided, this will replace the existing self.func.options (e.g. linear/log10 scale to estimate parameters)
        Returns
        -------
        UDEProblem
            A new instance of UDEProblem with the specified modifications.
        """
        dynamical_model = self.dynamical_model
        dynamic_parameters = new_dynamic_parameters if new_dynamic_parameters is not None \
            else self.func.dynamic_parameters
        options = new_options if new_options is not None \
            else self.func.options
        state_dimension = self.func.state_dimension
        mlp = new_mlp if new_mlp is not None else self.func.mlp
        regularise_network = new_regularise_network if new_regularise_network is not None \
            else self.func.regularise_network
        return UDEProblem(dynamical_model, dynamic_parameters, mlp, options, state_dimension, regularise_network)


    def get_hyperparams(self) -> dict[str, Any]:
        """
        Get UDEProblem (hyper)parameters.

        Note: used in model serialisation
        """
        # Cannot seri
        activation_fn_name = None
        final_activation_fn_name = None
        for key in activation_functions.keys():
            if self.func.mlp.activation == activation_functions[key]:
                activation_fn_name = key
            if self.func.mlp.final_activation == activation_functions[key]:
                final_activation_fn_name = key

        return {
            'dynamical_model': self.dynamical_model,
            'dynamic_parameters': np.array(self.func.dynamic_parameters).tolist(),
            'options': self.func.options,
            'state_dimension': self.func.state_dimension,
            'regularise_network': self.func.regularise_network,
            'in_size': self.func.mlp.in_size,
            'out_size': self.func.mlp.out_size,
            'width_size': self.func.mlp.width_size,
            'depth': self.func.mlp.depth,
            'use_bias' : self.func.mlp.use_bias,
            'use_final_bias': self.func.mlp.use_final_bias,
            'activation': activation_fn_name,
            'final_activation': final_activation_fn_name,
            'key': jr.PRNGKey(seed=0).tolist(),  # key is unimportant - parameters will be overwritten by serialised MLP
            'input_dropout_rate': self.func.mlp.input_dropout_rate
            if isinstance(self.func.mlp, CustomDropoutMLP) else None,
            'hidden_dropout_rate': self.func.mlp.hidden_dropout_rate
            if isinstance(self.func.mlp, CustomDropoutMLP) else None,
        }


    def save(self, filename: Path) -> None:
        """
        Save the model to a file.

        :param filename: path of file to serialise UDEFunction
        """
        # Serialise UDEFunction (and MLP)
        filename.parent.mkdir(exist_ok=True, parents=True)
        with Path.open(filename, 'wb') as f:
            # Save model hyperparameters
            hyperparam_str = json.dumps(self.get_hyperparams())
            f.write((hyperparam_str + '\n').encode())
            # Serialise MLP
            eqx.tree_serialise_leaves(f, self.func.mlp)


    @classmethod
    def load(
            cls,
            filename: Union[Path, str],
    ) -> 'UDEProblem':
        """
        Loads UDEProblem from serialised UDEFunction file.

        :param filename: path of file
        :return: UDEProblem instance
        """
        # Ensure filename is a Path object
        filename = Path(filename)
        with Path.open(filename, 'rb') as f:
            # Load model hyperparameters
            hyperparam_str = f.readline().decode().strip()
            hyperparams = json.loads(hyperparam_str)
            # Key: convert back to ArrayImpl with expected dtype
            hyperparams['key'] = jnp.array(hyperparams['key'], dtype=jnp.uint32)
            # Subset to MLP params
            mlp_hyperparams = {
                param_key: hyperparams[param_key]
                for param_key in [
                    "in_size", "out_size",
                    "width_size", "depth",
                    "use_bias", "use_final_bias",
                    "key"
                ]
            }
            mlp_skeleton_func = eqx.nn.MLP
            # Handle CustomDropoutMLP-specific params
            for param_key in ["input_dropout_rate", "hidden_dropout_rate"]:
                if hyperparams[param_key] is not None:
                    mlp_hyperparams[param_key] = hyperparams[param_key]
                    # Override default MLP function
                    mlp_skeleton_func = CustomDropoutMLP

            # Make MLP skeleton
            mlp = mlp_skeleton_func(
                **mlp_hyperparams,
                activation=activation_functions[hyperparams["activation"]],
                final_activation=activation_functions[hyperparams["final_activation"]]
            )
            # Apply serialised weights and biases to MLP skeleton
            mlp = eqx.tree_deserialise_leaves(f, mlp)
            ude_prob = UDEProblem(
                dynamical_model=hyperparams["dynamical_model"],
                dynamic_parameters=jnp.array(hyperparams["dynamic_parameters"]),  # from list to JAX Array
                mlp=mlp,
                options=hyperparams["options"],
                state_dimension=hyperparams["state_dimension"],
                regularise_network=hyperparams["regularise_network"],
            )
        return ude_prob


def build_mlp(
        in_size: int,
        out_size: int,
        width_size: int,
        hidden_layers: int,
        key: Any,
        use_bias: bool,
        use_final_bias: bool,
        activation: Callable,
        final_activation: Callable,
        use_dropout: bool = False,  # default: regular MLP (no dropout)
        input_dropout_rate: float = 0.0,
        hidden_dropout_rate: float = 0.0,
        use_skip_connections: bool = False,  # default: regular MLP (no residual connections)
) -> Union[eqx.nn.MLP, eqx.Module]:
    if not use_dropout:
    if not use_dropout and not use_skip_connections:
        mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=hidden_layers,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            activation=activation,
            final_activation=final_activation,
            key=key
        )
    elif use_dropout and (input_dropout_rate is not None) and (hidden_dropout_rate is not None):
        mlp = CustomDropoutMLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=hidden_layers,
            activation=activation,
            final_activation=final_activation,
            key=key,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            dropout_rates={
                "input": input_dropout_rate,
                "hidden": hidden_dropout_rate,
            },
        )
    elif use_skip_connections:
        mlp = ResidualMLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=hidden_layers,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            activation=activation,
            final_activation=final_activation,
            key=key
        )
    else:
        raise ValueError("For use_dropout to be True, dropout_rate cannot be None.")
    return mlp
