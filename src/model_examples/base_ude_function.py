import equinox as eqx
from jaxtyping import Array
from typing import Optional, Union


class BaseUDEFunction(eqx.Module):
    """Common initialisation logic for UDEFunction classes."""

    def max_stable_dt(self) -> Optional[Array]:
        """Largest step an explicit solver may take on this vector field, or None if unconstrained.

        Only models carrying a KNOWN stiff linear operator (e.g. a diffusion term with learnable
        coefficients) can state such a bound; everything else returns None and the adaptive
        controller is left to its own devices. See ``UDEProblem.cap_controller_dt``.
        """
        return None

    dynamic_parameters: Array
    mlp: Union[eqx.nn.MLP, eqx.Module]
    options: dict
    state_dimension: int
    state_delimiter: int
    regularise_network: bool

    def __init__(
            self,
            dynamic_parameters,
            mlp,
            options,
            state_dimension,
            state_delimiter,
            regularise_network,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dynamic_parameters = dynamic_parameters
        self.mlp = mlp
        self.options = options
        self.state_dimension = state_dimension
        self.state_delimiter = state_delimiter
        self.regularise_network = regularise_network if regularise_network is not None else False
