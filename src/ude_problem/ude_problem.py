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
from src.model_examples.lotka_volterra.setup_diffrax_model import UDEFunction as UDEFunction_lv
from src.model_examples.sir.setup_diffrax_model import UDEFunction as UDEFunction_sir
from src.ude_problem import CustomDropoutMLP, ResidualMLP
from typing import Any, Callable, List, Optional, Tuple, Union


UDE_funcs = {
    'lotka_volterra': UDEFunction_lv,
    'sir': UDEFunction_sir,
}


def initialize_ms(
        effective_state_dimension: int,
        shooting_window_length: int,
        training_horizon: int,
        t_save: diffrax.SaveAt,
        strategy: str,
        u0_guess: Optional[Tuple],
        network_regularisation: bool,
        curriculum_learning: bool = False,
        curriculum_params: Optional[dict[str, tuple]] = None,
        random_seed: int = 0,
) -> tuple[Tuple, Tuple, Array]:
    """
    Initialize auxiliary variables (num_windows, starting and ending points,
    save timepoints) and learnable initial values for the shooting windows.

    Parameters
    ----------
    effective_state_dimension: int
        The number of input features for the neural network. Includes state variables and augmentation variables.
    shooting_window_length: int
        The number of timepoints in each shooting window.
    training_horizon: int
        Overall training horizon (time-range of training datapoints).
    t_save: diffrax.SaveAt
        Timepoints at which to save the training simulation.
    strategy: str
        Initialization strategy for initial conditions.
    u0_guess: Tuple
        Initial guess for the initial conditions.
    network_regularisation: bool
        If True, the network contribution to the dynamics will be regularised.
    curriculum_learning: bool
        If True, the model will be trained using curriculum learning.
    curriculum_params: dict
        Timepoints and masks for first curriculum learning stage.
    random_seed: int
        Random seed for random initialization.
    """
    # Pre-compute number of windows
    num_windows = int(np.ceil(training_horizon / (shooting_window_length - 1)))

    if curriculum_learning and (curriculum_params is not None):
        # pre-computed
        t_save_windows, masks = curriculum_params["time"], curriculum_params["mask"]
    elif curriculum_learning and (curriculum_params is None):
        raise ValueError("curriculum_params must be provided if curriculum_learning is True!")
    else:
        t_save_windows, masks = generate_ms_time_and_mask(
            num_windows, shooting_window_length, training_horizon, t_save
        )

    if strategy == 'z':
        # Augment initial conditions to include network regularisation
        if network_regularisation:
            effective_state_dimension += 1
        return t_save_windows, masks, jnp.zeros((num_windows, effective_state_dimension))
    # TODO @GiacomoFabrini add an option to initialise with observed initial conditions
    elif strategy == 'g':
        if network_regularisation:
            u0_guess = jnp.concatenate([jnp.array(u0_guess), jnp.zeros(1)])
        return t_save_windows, masks, jnp.tile(u0_guess, (num_windows, 1))
    elif strategy == 'ru':
        # Sample uniformly state_dimension variables
        u0 = jr.uniform(jr.PRNGKey(random_seed), (num_windows, effective_state_dimension))
        # Concatenate a 0 across all num_windows if using network regularisation
        if network_regularisation:
            u0 = jnp.concatenate([u0, jnp.zeros((num_windows, 1))], axis=1)
        # return u0
        return t_save_windows, masks, u0
    else:
        raise ValueError(f"Invalid shooting_iv_init strategy: {strategy}")


def initialize_single(
    training_horizon: int,
    t_save: diffrax.SaveAt,
    u0_guess: Tuple,
    curriculum_learning: bool,
    curriculum_params: Optional[dict[str, tuple]] = None
) -> tuple[Tuple, Tuple, Array]:
    if curriculum_learning:
        if curriculum_params is not None:
            t_windows = tuple(jnp.expand_dims(jnp.array(curriculum_params["time"]), axis=0).tolist())
            masks = tuple(jnp.expand_dims(jnp.array(curriculum_params["mask"]), axis=0).tolist())
        else:
            raise ValueError("curriculum_indices must be provided if curriculum_learning is True")
    else:
        t_windows = tuple(jnp.array([t_save.subs.ts]).tolist())
        masks = tuple(jnp.array([jnp.ones(training_horizon)]).tolist())
    u0 = jnp.expand_dims(jnp.array(u0_guess), axis=0)
    return t_windows, masks, u0


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
    func: Union[UDEFunction_lv, UDEFunction_sir]
    training_horizon: int
    shooting_mode: str
    shooting_window_length: Optional[int]
    shooting_iv_init: Optional[str]
    t_save_windows: Optional[Tuple]
    masks: Optional[Tuple]
    initial_values: Optional[Array]
    u0_guess: Optional[Tuple]
    init_random_seed: Optional[int]
    curriculum_learning: bool
    curriculum_params: Optional[dict[str, tuple]]
    curriculum_stage: Optional[int]

    def __init__(
            self, dynamical_model, dynamic_parameters, mlp, options, state_dimension, training_horizon,
            shooting_mode: str = "single", shooting_window_length=10, shooting_iv_init="zeros",
            u0_guess=None, init_random_seed=0, regularise_network=None,
            t_save_train: Optional[diffrax.SaveAt] = None,
            initial_values: Optional[Array] = None, masks: Optional[Array] = None, t_save_windows: Optional[Array] = None,
            curriculum_learning: bool = False, curriculum_params: Optional[dict[str, tuple]]=None,
            curriculum_stage: Optional[int]=1, **kwargs,
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
        training_horizon: int
            The length of the training horizon.
        shooting_mode: str
            The shooting mode to use. Can be 'single' or 'multiple'.
        shooting_window_length: int
            The length of the shooting windows.
        shooting_iv_init: str
            The initialisation strategy for the shooting windows.
        u0_guess: tuple
            Initial guess for the initial conditions.
        init_random_seed: int
            Random seed for random initialisation (if shooting_iv_init == "random_uniform").
        regularise_network: bool
            If True, the network contribution to the dynamics will be regularised.
        t_save_train: array-like
            Timepoints at which to save the training simulation.
        initial_values: array-like
            Pre-computed initial values for the shooting windows.
        t_save_windows: array-like
            Pre-computed timepoints for the shooting windows.
        curriculum_learning: bool
            If True, the model will be trained using curriculum learning.
        curriculum_params: dict
            Dictionary containing timepoints and masks for specific curriculum learning stage.
        curriculum_stage: int
            Current stage of CL, if applicable.
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
        self.curriculum_learning = curriculum_learning
        self.curriculum_params = curriculum_params
        if self.shooting_mode == "single":
            self.shooting_iv_init, self.init_random_seed, self.curriculum_stage = None, None, 1
            self.u0_guess = u0_guess
            if all(x is not None for x in (t_save_windows, initial_values, masks)):
                # TODO @Giacomo - FIX REPETITION!
                self.t_save_windows, self.masks, self.initial_values = t_save_windows, masks, initial_values
                # Augment initial values to include network regularisation (each u0 should have mlp.layers[0].in_features+1 elements)
                if self.func.regularise_network and (self.initial_values.shape[1] == self.func.mlp.layers[0].in_features):
                    self.initial_values = jnp.concatenate(
                        [self.initial_values, jnp.zeros((self.initial_values.shape[0], 1))], axis=1
                    )
            elif t_save_train is not None:
                self.t_save_windows, self.masks, self.initial_values = initialize_single(
                    training_horizon=self.training_horizon,
                    t_save=t_save_train,
                    u0_guess=self.u0_guess,
                    curriculum_learning=curriculum_learning,
                    curriculum_params=curriculum_params
                )
            else:
                raise ValueError("Either initial_values and t_save_windows OR t_save_train must be provided!")
        elif self.shooting_mode == "multiple":
            self.shooting_iv_init = shooting_iv_init
            # Initialise the learnable initial values for the shooting windows
            self.u0_guess = u0_guess
            self.init_random_seed = init_random_seed
            if all(x is not None for x in (t_save_windows, initial_values, masks, curriculum_stage)):
                # TODO @Giacomo - check whether there are other scenarios to be considered
                self.t_save_windows, self.masks, self.initial_values = t_save_windows, masks, initial_values
                self.curriculum_stage = curriculum_stage
                # Augment initial values to include network regularisation (each u0 should have mlp.layers[0].in_features+1 elements)
                if self.func.regularise_network and (self.initial_values.shape[1] == self.func.mlp.layers[0].in_features):
                    self.initial_values = jnp.concatenate(
                        [self.initial_values, jnp.zeros((self.initial_values.shape[0], 1))], axis=1
                    )
            elif t_save_train is not None:
                self.t_save_windows, self.masks, self.initial_values = initialize_ms(
                    effective_state_dimension=int(self.func.mlp.layers[0].in_features),
                    shooting_window_length=self.shooting_window_length,
                    training_horizon=self.training_horizon,
                    t_save=t_save_train,
                    strategy=self.shooting_iv_init,
                    u0_guess=self.u0_guess,
                    network_regularisation=regularise_network,
                    random_seed=self.init_random_seed,
                    curriculum_learning=curriculum_learning,
                    curriculum_params=curriculum_params
                )
                # Initialise to first CL stage
                self.curriculum_stage = 1
            else:
                raise ValueError("Either initial_values and t_save_windows OR t_save_train must be provided for MS!")
        else:
            # TODO FIX THIS - WINDOW_BATCH
            self.shooting_iv_init, self.u0_guess, self.init_random_seed = None, None, None
            self.initial_values, self.masks, self.t_save_windows = None, None, None
            self.curriculum_stage = None


    def get_augmented_state_variables(self):
        return self.initial_values[:, self.func.state_dimension:self.func.mlp.layers[0].in_features][0]

    def get_auxiliary_output_variable(self):
        return self.initial_values[:, self.func.mlp.layers[0].in_features:jnp.newaxis][0]


    def solve_single_shooting(
            self,
            solver: diffrax.AbstractSolver,
            u0: Array,
            t_save: Array,
            adjoint: diffrax.AbstractAdjoint = diffrax.RecursiveCheckpointAdjoint(),
            controller=diffrax.PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=1e-8, atol=1e-8),
    ):
        """
        Solve the UDE using single-shooting (or single-shooting + CL) for training.
        u0 and t_save can be the same as the internal initial values and timepoints, respectively.
        """
        t0, t1 = [t_save[0], t_save[-1]]
        sol = diffrax.diffeqsolve(diffrax.ODETerm(self.func),
                                  solver,
                                  t0,
                                  t1,
                                  None,
                                  y0=jnp.concatenate(
                                      [
                                          jax.lax.stop_gradient(u0[:self.func.state_dimension]),
                                          self.get_augmented_state_variables(),  # only update initial value for augmentation vars
                                          jax.lax.stop_gradient(self.get_auxiliary_output_variable()),
                                      ],
                                      axis=0
                                  ),
                                  args=u0,
                                  saveat=diffrax.SaveAt(ts=t_save),
                                  throw=False,
                                  stepsize_controller=controller,
                                  adjoint=adjoint)
        return sol


    def solve_single_shooting_validation(
            self,
            solver: diffrax.AbstractSolver,
            u0,
            t_save: diffrax.SaveAt,
            t_span: List,
            adjoint: diffrax.AbstractAdjoint = diffrax.RecursiveCheckpointAdjoint(),
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
            None,
            y0=jnp.concatenate(
                [
                    u0[:self.func.state_dimension],
                    self.get_augmented_state_variables(),  # insert trained, internal augmentation variables
                    self.get_auxiliary_output_variable(),
                ],
                axis=0
            ),
            args=u0,
            saveat=t_save,
            throw=False,
            stepsize_controller=controller,
            adjoint=adjoint
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
        """

        solution = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(self.func),
            solver=solver,
            t0=t_save[0],
            t1=t_save[-1],
            dt0=None,
            y0=jnp.concatenate(
                [
                    u0[:self.func.mlp.layers[0].in_features],  # state vars + potential augmentation - learnable
                    jax.lax.stop_gradient(u0[-1:jnp.newaxis]),  # do NOT update network regularisation IV
                    # u0[-1:jnp.newaxis],  # alternative, NOT USED: update network regularisation initial value
                ],
                axis=0
            ),
            saveat=diffrax.SaveAt(ts=t_save),
            stepsize_controller=controller,
            adjoint=adjoint
        )

        return solution


    def update_curriculum_time_mask(self, time: Tuple, mask: Tuple):
        new_time = jnp.expand_dims(jnp.array(time), axis=0) if len(jnp.array(time).shape) == 1 else jnp.array(time)
        new_mask = jnp.expand_dims(jnp.array(mask), axis=0) if len(jnp.array(mask).shape) == 1 else jnp.array(mask)
        time_update = eqx.tree_at(
            lambda m: m.t_save_windows,
            self,
            tuple(new_time.tolist())
        )
        return eqx.tree_at(
            lambda m: m.masks,
            time_update,
            tuple(new_mask.tolist())
        )


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


    def zero_last_initial_values(self):
        """
        Zeroes the last initial values in the UDEProblem.
        """
        return eqx.tree_at(
            lambda m: m.initial_values,
            self,
            jnp.concatenate(
                [
                    self.initial_values[:-(self.curriculum_stage - 1)],
                    jnp.zeros((self.curriculum_stage - 1, self.initial_values.shape[1]))
                ],
                axis=0
            )
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


    def advance_curriculum_stage(self):
        """
        Advances the curriculum learning stage by one.
        """
        return eqx.tree_at(
            lambda m: m.curriculum_stage,
            self,
            self.curriculum_stage + 1
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
            'training_horizon': self.training_horizon,
            'shooting_mode': self.shooting_mode,
            'shooting_window_length': self.shooting_window_length,
            'shooting_iv_init': self.shooting_iv_init,
            'u0_guess': np.array(self.u0_guess).tolist() if self.u0_guess is not None else None,
            'initial_values': np.array(
                self.initial_values
            ).tolist() if self.initial_values is not None else None,
            'masks': np.array(
                self.masks
            ).tolist() if self.masks is not None else None,
            't_save_windows': np.array(
                self.t_save_windows
            ).tolist() if self.t_save_windows is not None else None,
            'init_random_seed': self.init_random_seed,
            'regularise_network': self.func.regularise_network,
            'curriculum_learning': self.curriculum_learning,
            'curriculum_params': {
                'time': np.array(self.curriculum_params["time"]).tolist() if self.curriculum_params is not None else None,
                'mask': np.array(self.curriculum_params["mask"]).tolist() if self.curriculum_params is not None else None
            },
            "curriculum_stage": self.curriculum_stage if self.curriculum_learning else None,
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
                u0_guess=jnp.array(hyperparams["u0_guess"], dtype=jnp.float64),
                init_random_seed=hyperparams["init_random_seed"],
                regularise_network=hyperparams["regularise_network"],
                # Multiple-shooting attributes
                initial_values=jnp.array(hyperparams["initial_values"], dtype=jnp.float64),
                masks=jnp.array(hyperparams["masks"], dtype=jnp.float64),
                t_save_windows=jnp.array(hyperparams["t_save_windows"], dtype=jnp.float64),
                # Curriculum learning attributes
                curriculum_learning=hyperparams["curriculum_learning"],
                curriculum_params={
                    "time": tuple(hyperparams["curriculum_params"]["time"]),
                    "mask": tuple(hyperparams["curriculum_params"]["mask"]),
                },
                curriculum_stage=hyperparams["curriculum_stage"],
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


def generate_ms_time_and_mask(
        num_windows: int,
        shooting_window_length: int,
        training_horizon: int,
        t_save: diffrax.SaveAt
) -> Tuple[Tuple, Tuple]:
    # Compute auxiliary MS variables
    t_windows, masks = [], []
    for i in range(num_windows):
        # Select the core timepoints
        start_idx = i * (shooting_window_length - 1)
        end_idx = (i + 1) * shooting_window_length - i
        core_indices = slice(start_idx, end_idx)
        core_values = t_save.subs.ts[core_indices]

        # Construct full-length window
        window = np.zeros(training_horizon)
        window[: start_idx] = core_values[0]
        window[core_indices] = core_values
        window[end_idx:] = core_values[-1]
        t_windows.append(window)

        # Construct mask
        mask = np.zeros(training_horizon)
        mask[core_indices] = 1
        masks.append(mask)

    t_save_windows = tuple(jnp.array(t_windows).tolist())
    masks = tuple(jnp.array(masks).tolist())

    # import matplotlib.pyplot as plt
    # for arr, label in zip([t_save_windows, masks], ["time", "mask"]):
    #     plt.imshow(arr, vmin=0, vmax=1)
    #     ax = plt.gca()
    #     ax.yaxis.set_visible(False)  # Hide y-axis labels and ticks
    #     plt.savefig(f"multiple.{shooting_window_length}.noCL.{label}.pdf")
    return t_save_windows, masks
