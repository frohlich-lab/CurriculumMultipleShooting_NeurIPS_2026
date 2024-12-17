import diffrax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import json
import numpy as np

from pathlib import Path

from jaxtyping import Array
from src.common import activation_functions
from src.model_examples.lotka_volterra.setup_diffrax_model import UDEFunction as UDEFunction_lv
from src.model_examples.sir.setup_diffrax_model import UDEFunction as UDEFunction_sir
from src.ude_problem import CustomDropoutMLP, ResidualMLP
from typing import Any, Callable, Optional, Union


UDE_funcs = {
    'lotka_volterra': UDEFunction_lv,
    'sir': UDEFunction_sir,
}


def initialize_shooting_iv(
        state_dimension: int,
        num_windows: int,
        strategy: str,
        u0_guess: Optional[Array],
        network_regularisation: bool
):
    """
    Initialise the learnable initial values for the shooting windows.

    Parameters
    ----------
    state_dimension: int
        The dimensionality of the system state.
    num_windows: int
        The number of shooting windows.
    strategy: str
        Initialization strategy for initial conditions.
    u0_guess: array-like
        Initial guess for the initial conditions.
    network_regularisation: bool
        If True, the network contribution to the dynamics will be regularised.
    """
    if strategy == 'zeros':
        # Augment initial conditions to include network regularisation
        effective_state_dim = state_dimension+1 if network_regularisation else state_dimension
        return jnp.zeros((num_windows, effective_state_dim))
    # TODO @GiacomoFabrini add an option to initialise with observed initial conditions
    elif strategy == 'guess':
        if network_regularisation:
            u0_guess = jnp.concatenate([u0_guess, jnp.zeros(1)])
        return jnp.tile(u0_guess, (num_windows, 1))
    else:
        raise ValueError(f"Invalid shooting_iv_init strategy: {strategy}")


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
    training_horizon: int
    shooting_mode: str
    shooting_window_length: Optional[int]
    shooting_iv_init: str
    initial_values: Optional[Array]
    u0_guess: Optional[Array]

    def __init__(
            self, dynamical_model, dynamic_parameters, mlp, options, state_dimension, training_horizon,
            shooting_mode: str = "single", shooting_window_length=10, shooting_iv_init="zeros", u0_guess=None,
            regularise_network=None, **kwargs
    ):
        """
        Parameters
        ----------
        dynamical_model: str
            The name of the dynamical model to use. Currently only 'lotka_volterra' and 'sir'.
        dynamic_parameters : jax.Array
            Starting values for the model dynamic parameters (α, δ in the model
            equations).
        mlp : eqx.nn.MLP
            An equinox Standard Multi-Layer Perceptron (MLP) network. If set to
            None path_meta must be provided.
        options: dict
            Options for the UDEFunction.
        state_dimension: int
            The number of state variables in the dynamical model.
        regularise_network: bool
            If True, the network contribution to the dynamics will be regularised.
        """
        super().__init__(**kwargs)
        self.dynamical_model = dynamical_model
        if self.dynamical_model not in UDE_funcs.keys():
            raise ValueError(f"Unknown dynamical model: {dynamical_model}!")
        self.func = UDE_funcs[dynamical_model](dynamic_parameters, mlp, options, state_dimension, regularise_network)
        self.training_horizon = training_horizon
        self.shooting_mode = shooting_mode
        if self.shooting_mode == "multiple" and shooting_window_length is None:
            raise ValueError("shooting_window_length cannot be None for shooting_mode='multiple'")
        # Conditionally define the shooting window length and compute the number of shooting windows
        self.shooting_window_length = shooting_window_length if self.shooting_mode == "multiple" else self.training_horizon
        num_windows = int(np.ceil(self.training_horizon / (self.shooting_window_length - 1))) \
            if self.shooting_mode == "multiple" else 1
        self.shooting_iv_init = shooting_iv_init
        # Initialise the learnable initial values for the shooting windows
        self.u0_guess = u0_guess
        self.initial_values = initialize_shooting_iv(
            state_dimension, num_windows, self.shooting_iv_init, self.u0_guess, regularise_network
        )


    def solve(self, solver, u0, t_save, t_span, *args):
        if self.shooting_mode == "single":
            return self.solve_single_shooting(solver, u0, t_save, t_span, *args)
        elif self.shooting_mode == "multiple":
            return self.solve_multiple_shooting(solver, u0, t_save, t_span, *args)


    def solve_single_shooting(self, solver, u0, t_save, t_span,
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

    def solve_multiple_shooting(
            self, solver, u0, t_save, t_span,
            controller=diffrax.PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=1e-8, atol=1e-8),
            adjoint=diffrax.RecursiveCheckpointAdjoint()
    ):
        """
        Solve the UDE using multiple shooting.

        Parameters
        ----------
        solver: diffrax.AbstractSolver
            The ODE solver to use.
        u0: array-like
            The initial values for the ODE // not used for MS for now.
        t_save: array-like
            The time points at which to save the solution.
        t_span: tuple
            Start and end times for the integration // not used for MS for now.
        controller: diffrax.PIDController, optional
            Adaptive stepsize controller.
        adjoint: diffrax.AbstractAdjoint, optional
            Adjoint method for backpropagation.
        """
        if not isinstance(t_save, diffrax.SaveAt):
            t_save = diffrax.SaveAt(ts=t_save)
        num_windows = int(np.ceil(len(t_save.subs.ts) / (self.shooting_window_length - 1)))
        solutions = []

        for i in range(num_windows):
            # Time range for the current shooting window
            window_start = i * (self.shooting_window_length - 1)
            window_end = window_start + self.shooting_window_length

            t_window = t_save.subs.ts[window_start:window_end]
            u0_window = self.initial_values[i]

            # Solve for the current window
            t_save_window = diffrax.SaveAt(ts=t_window)
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(self.func),
                solver,
                t0=t_window[0],
                t1=t_window[-1],
                dt0=None,
                y0=u0_window,
                saveat=t_save_window,
                stepsize_controller=controller,
                adjoint=adjoint
            )
            solutions.append(sol)

        return solutions

    def _solve_error(self):
        raise ValueError(f"Invalid shooting mode: {self.shooting_mode}")


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
        training_horizon = self.training_horizon
        shooting_mode = self.shooting_mode
        shooting_window_length = self.shooting_window_length
        shooting_iv_init = self.shooting_iv_init
        u0_guess = self.u0_guess
        regularise_network = new_regularise_network if new_regularise_network is not None \
            else self.func.regularise_network
        return UDEProblem(
            dynamical_model, dynamic_parameters, mlp, options, state_dimension,
            training_horizon, shooting_mode, shooting_window_length, shooting_iv_init, u0_guess, regularise_network
        )


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
            'options': {
                'dynamic_params_scale': self.func.options["dynamic_params_scale"],
                'bounded_tanh_params': {
                    key: np.array(item).tolist()  # cannot JSON-serialise JAX arrays
                    for key, item in self.func.options["bounded_tanh_params"].items()
                }
            },
            'state_dimension': self.func.state_dimension,
            'training_horizon': self.training_horizon,
            'shooting_mode': self.shooting_mode,
            'shooting_window_length': self.shooting_window_length,
            'shooting_iv_init': self.shooting_iv_init,
            'u0_guess': np.array(self.u0_guess).tolist() if self.u0_guess is not None else None,
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
                options={
                    "dynamic_params_scale": hyperparams["options"]["dynamic_params_scale"],
                    "bounded_tanh_params": {
                        key: jnp.array(item)
                        for key, item in hyperparams["options"]["bounded_tanh_params"].items()
                    }
                },
                state_dimension=hyperparams["state_dimension"],
                training_horizon=hyperparams["training_horizon"],
                shooting_mode=hyperparams["shooting_mode"],
                shooting_window_length=hyperparams["shooting_window_length"],
                shooting_iv_init=hyperparams["shooting_iv_init"],
                u0_gues=jnp.array(hyperparams["u0_guess"], dtype=jnp.float64),
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
