import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import json
import numpy as np

from pathlib import Path

from jaxtyping import Array
from src.common import activation_functions
from src.model_examples.lotka_volterra.setup_lotka_volterra_model import LotkaVolterraUDEFunction
from src.model_examples.sir.setup_sir_model import SIRUDEFunction
from src.model_examples.pendulum.setup_pendulum_model import PendulumNODEFunction
from src.model_examples.reaction_diffusion.setup_reaction_diffusion_model import (
    REACTION_DIFFUSION_SIZES,
    ReactionDiffusionNODEFunction,
    ReactionDiffusionUDEFunction,
)
from src.ude_problem import CustomDropoutMLP
from typing import Any, Callable, List, Optional, Union


UDE_funcs = {
    'lotka_volterra': LotkaVolterraUDEFunction,
    'sirKorea': SIRUDEFunction,
    'sirUK': SIRUDEFunction,
    'pendulum': PendulumNODEFunction,
}
for _rd_N in REACTION_DIFFUSION_SIZES:
    UDE_funcs[f"reaction_diffusion_ude_{_rd_N}"] = ReactionDiffusionUDEFunction
    UDE_funcs[f"reaction_diffusion_node_{_rd_N}"] = ReactionDiffusionNODEFunction


class UDEProblem(eqx.Module):
    """
    Class to solve the UDEProblem with a chosen ODE-solver

    Methods
    -------
    solve(self, solver t_save, t_span)
        Solve the UDE for t_span = [t0, tend] and save the solution at the
        points in t_save using the provided solver. Controller is the adaptive
        ODE-stepping controller (should not need to be changed).
    """
    dynamical_model: str
    func: Union[LotkaVolterraUDEFunction, SIRUDEFunction]
    initial_values: Array
    scale_initial_values: Callable
    descale_initial_values: Callable

    def __init__(
            self, dynamical_model, dynamic_parameters, mlp, options, state_dimension,
            initial_values: Array, regularise_network=None, scale_initial_values=None, **kwargs,
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
        scale_initial_values: Callable, optional
            Function to scale the initial values appropriately for the UDEFunction.
        initial_values: array-like
            Pre-computed initial values for the shooting windows.
        """
        super().__init__(**kwargs)
        self.dynamical_model = dynamical_model
        if self.dynamical_model not in UDE_funcs.keys():
            raise ValueError(f"Unknown dynamical model: {dynamical_model}!")
        if self.dynamical_model == "lotka_volterra":
            state_delimiter = mlp.layers[0].in_features
        elif self.dynamical_model.startswith("sir"):
            state_delimiter = mlp.layers[0].in_features + 1
        elif self.dynamical_model == "pendulum":
            state_delimiter = mlp.layers[0].in_features
        elif self.dynamical_model.startswith("reaction_diffusion"):
            # ConvNet (no `.layers[0].in_features`): the NN consumes `mlp.out_size` grid channels
            # (2 observed + node_augmentation), each of size N*N.
            _rd_N = int(self.dynamical_model.rsplit("_", 1)[1])
            state_delimiter = mlp.out_size * _rd_N * _rd_N
        else:
            raise ValueError(f"Unsupported dynamical model: {dynamical_model}!")
        self.func = UDE_funcs[self.dynamical_model](
            dynamic_parameters, mlp, options, state_dimension, state_delimiter, regularise_network
        )
        if not regularise_network:
            self.initial_values = initial_values
        else:
            if (
                    (dynamical_model == "lotka_volterra")
                    and (initial_values.shape[1] == self.func.mlp.layers[0].in_features)
            ) or (
                    (dynamical_model.startswith("sir"))
                    and (initial_values.shape[1] == self.func.mlp.layers[0].in_features + 1)
            ) or (
                (dynamical_model == "pendulum")
                and (initial_values.shape[1] == self.func.mlp.layers[0].in_features)
            ) or (
                (dynamical_model.startswith("reaction_diffusion"))
                and (initial_values.shape[1] == self.func.state_delimiter)
            ):
                # Add zero initial value for neural network contribution regularisation
                self.initial_values = jnp.concatenate(
                    [initial_values, jnp.zeros((initial_values.shape[0], 1))],
                    axis=1
                )
            else:
                # Loading from serialised model with regularisation IV already present
                self.initial_values = initial_values
        if scale_initial_values is None:
            self.scale_initial_values = lambda x: x
            self.descale_initial_values = lambda x: x
        else:
            # Support named scaling methods (strings) for robust, numerically-stable transforms
            def get_scale_descale(method: Union[str, Callable]):
                # Safe log10 scaling: clip input to avoid log(0) and negatives
                if method == "log10" or method is jnp.log10:
                    def scale_fn(x):
                        # Support 1D or 2D inputs: apply clipping then log10
                        squeezed = False
                        x_arr = jnp.asarray(x)
                        if x_arr.ndim == 1:
                            x_arr = x_arr[jnp.newaxis, :]
                            squeezed = True
                        out = jnp.log10(jnp.clip(x_arr, a_min=1e-12))
                        return out[0] if squeezed else out

                    def descale_fn(x):
                        # Clip logits to avoid overflow when exponentiating
                        x_arr = jnp.asarray(x)
                        squeezed = False
                        if x_arr.ndim == 1:
                            x_arr = x_arr[jnp.newaxis, :]
                            squeezed = True
                        x_clip = jnp.clip(x_arr, a_min=-12.0, a_max=12.0)
                        out = jnp.clip(10 ** x_clip, a_min=1e-12, a_max=1e12)
                        return out[0] if squeezed else out

                    return scale_fn, descale_fn
                # If a callable was supplied without a matching descaler, assume user manages both
                elif callable(method):
                    return method, None
                else:
                    raise NotImplementedError(f"Descaling not implemented for method: {method}")

            scale_fn, descale_fn = get_scale_descale(scale_initial_values)

            if regularise_network:
                # Only scale state/augmented columns, keep network-regularisation column unchanged
                def scale_with_reg(x):
                    x_arr = jnp.asarray(x)
                    squeezed = False
                    if x_arr.ndim == 1:
                        x_arr = x_arr[jnp.newaxis, :]
                        squeezed = True
                    scaled = scale_fn(x_arr[:, :-1])
                    out = jnp.concatenate([scaled, x_arr[:, -1:]], axis=1)
                    return out[0] if squeezed else out

                self.scale_initial_values = scale_with_reg
                if descale_fn is not None:
                    def descale_with_reg(x):
                        x_arr = jnp.asarray(x)
                        squeezed = False
                        if x_arr.ndim == 1:
                            x_arr = x_arr[jnp.newaxis, :]
                            squeezed = True
                        descaled = descale_fn(x_arr[:, :-1])
                        out = jnp.concatenate([descaled, x_arr[:, -1:]], axis=1)
                        return out[0] if squeezed else out

                    self.descale_initial_values = descale_with_reg
                else:
                    self.descale_initial_values = lambda x: x
            else:
                self.scale_initial_values = scale_fn
                self.descale_initial_values = descale_fn if descale_fn is not None else (lambda x: x)

            self.initial_values = self.scale_initial_values(self.initial_values)


    def get_internal_augmented_vars(self):
        return self.initial_values[:, self.func.state_dimension:self.func.state_delimiter][0]

    def get_auxiliary_output_reg_var(self):
        return self.initial_values[:, self.func.state_delimiter:jnp.newaxis][0]


    def solve_single_shooting(
            self,
            solver: diffrax.AbstractSolver,
            u0: Array,
            t_save: Array,
            adjoint: diffrax.AbstractAdjoint = diffrax.RecursiveCheckpointAdjoint(),
            max_steps: int = 20000,
            controller=diffrax.PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=1e-8, atol=1e-8),
    ):
        """
        Solve the UDE using single-shooting (or single-shooting + CL) for training.
        u0 and t_save can be the same as the internal initial values and timepoints, respectively.
        """
        t0, t1 = [t_save[0], t_save[-1]]
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            solver,
            t0,
            t1,
            1e-6,
            y0=u0,
            saveat=diffrax.SaveAt(ts=t_save),
            throw=False,
            stepsize_controller=controller,
            adjoint=adjoint,
            max_steps=max_steps
        )
        return sol


    def solve_single_shooting_validation(
            self,
            solver: diffrax.AbstractSolver,
            u0,
            t_save: diffrax.SaveAt,
            t_span: List,
            adjoint: diffrax.AbstractAdjoint = diffrax.RecursiveCheckpointAdjoint(),
            max_steps: int = 20000,
            controller=diffrax.PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=1e-8, atol=1e-8),
    ):
        """
        Solve the UDE using single-shooting for comparison with validation data.
        As this will not be wrapped in an eqx.filter_value_and_grad, we don't need to worry about stopping
        gradients for various u0 components.
        """

        t0, t1 = t_span
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            solver,
            t0,
            t1,
            1e-6,
            y0=jnp.concatenate(
                [
                    u0[:self.func.state_dimension],
                    self.get_internal_augmented_vars(),  # plug in the trained augmentation variables
                    u0[-1:],
                ],
                axis=0
            ),
            saveat=t_save,
            throw=False,
            stepsize_controller=controller,
            adjoint=adjoint,
            max_steps=max_steps
        )
        return sol


    @eqx.filter_vmap(
        in_axes={
            "self": None,
            "solver": None,
        }
    )
    def solve_multiple_shooting(
            self,
            solver: diffrax.AbstractSolver,
            u0: Array,
            t_save: Array,
            adjoint: diffrax.AbstractAdjoint = diffrax.RecursiveCheckpointAdjoint(),
            max_steps: int = 20000,
            controller=diffrax.PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=1e-8, atol=1e-8),
    ):
        """
        Solve the UDE using multiple shooting.

        Parameters
        ----------
        solver: diffrax.AbstractSolver
            The ODE solver to use.
        u0: array-like
            The initial values for the ODE - internal values passed from outside vmap.
        t_save: array-like
            The time points at which to save the solution - internal values passed from outside vmap.
        controller: diffrax.PIDController, optional
            Adaptive stepsize controller.
        adjoint: diffrax.AbstractAdjoint, optional
            Adjoint method for backpropagation.
        max_steps: int, optional
            Maximum number of steps for diffrax.diffeqsolve().
        """
        solution = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(self.func),
            solver=solver,
            t0=t_save[0],
            t1=t_save[-1],
            dt0=1e-6,
            y0=u0,
            saveat=diffrax.SaveAt(ts=t_save),
            throw=False,
            stepsize_controller=controller,
            adjoint=adjoint,
            max_steps=max_steps
        )

        return solution


    def replace_mlp(
            self,
            new_mlp: Union[CustomDropoutMLP, eqx.nn.MLP],
            regularise_network: Optional[bool] = None,
            switch_to_inference: bool = False
    ):
        """
        Replaces the MLP in the UDEFunction with a new one.
        Parameters
        ----------
        new_mlp:
            The new MLP to use.
        regularise_network: Optional[bool] (default None)
            If not None, regularise_network flag will be set to this value.
        switch_to_inference: bool (default False)
            If True, MLP will be set to inference mode (dropout disabled).

        Returns
        -------
        UDEProblem
            A new instance of UDEProblem with the specified modifications.
        """
        new_model = eqx.tree_at(
            lambda m: m.func.mlp,
            self,
            new_mlp if not switch_to_inference else eqx.nn.inference_mode(new_mlp, value=True)
        )
        if regularise_network is not None:
            new_model = eqx.tree_at(
                lambda m: m.func.regularise_network,
                new_model,
                regularise_network
            )
            new_model = eqx.tree_at(
                lambda m: m.initial_values,
                new_model,
                jnp.concatenate([self.initial_values, jnp.zeros((self.initial_values.shape[0], 1))], axis=1)
            )
        return new_model


    def replace_dynamic_parameters(self, new_dynamic_parameters: jnp.ndarray):
        """
        Replaces the dynamic parameters in the UDEFunction with new ones.
        Parameters
        ----------
        new_dynamic_parameters: jnp.ndarray
            The new dynamic parameters to use.

        Returns
        -------
        UDEProblem
            A new instance of UDEProblem with the specified modifications.
        """
        assert isinstance(new_dynamic_parameters, jnp.ndarray), \
            f"New dynamic parameters must be a jnp.ndarray, not {type(new_dynamic_parameters)}"
        assert new_dynamic_parameters.shape == self.func.dynamic_parameters.shape, \
            (f"New dynamic parameters shape {new_dynamic_parameters.shape} does not match "
             f"existing shape {self.func.dynamic_parameters.shape}!")
        return eqx.tree_at(
            lambda m: m.func.dynamic_parameters,
            self,
            new_dynamic_parameters
        )


    def replace_options(self, new_options: dict):
        """
        Replaces the options in the UDEFunction with new ones.
        Parameters
        ----------
        new_options: dict
            The new options to use.

        Returns
        -------
        UDEProblem
            A new instance of UDEProblem with the specified modifications.
        """
        assert isinstance(new_options, dict), f"New options must be a dict, not {type(new_options)}"
        return eqx.tree_at(
            lambda m: m.func.options,
            self,
            new_options
        )


    def get_hyperparams(self) -> dict[str, Any]:
        """
        Get UDEProblem (hyper)parameters.

        Note: used in model serialisation
        """
        # Cannot serialise activation functions directly - use string name
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
            'initial_values': np.array(
                self.initial_values
            ).tolist() if self.initial_values is not None else None,
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
                        key: tuple(item)
                        for key, item in hyperparams["options"]["bounded_tanh_params"].items()
                    }
                },
                state_dimension=hyperparams["state_dimension"],
                regularise_network=hyperparams["regularise_network"],
                initial_values=jnp.array(hyperparams["initial_values"], dtype=jnp.float64),
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
) -> Union[eqx.nn.MLP, eqx.Module]:
    if not use_dropout:
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
    else:
        raise ValueError("For use_dropout to be True, dropout_rate cannot be None.")
    return mlp
