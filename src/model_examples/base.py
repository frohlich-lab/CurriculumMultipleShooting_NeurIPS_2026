import equinox as eqx
from jaxtyping import Array
from typing import Union


class BaseUDEFunction(eqx.Module):
    """Common initialisation logic for UDEFunction classes."""

    dynamic_parameters: Array
    mlp: Union[eqx.nn.MLP, eqx.Module]
    options: dict
    state_dimension: int
    regularise_network: bool

    def __init__(self, dynamic_parameters, mlp, options, state_dimension, regularise_network, **kwargs):
        super().__init__(**kwargs)
        self.dynamic_parameters = dynamic_parameters
        self.mlp = mlp
        self.options = options
        self.state_dimension = state_dimension
        self.regularise_network = regularise_network if regularise_network is not None else False
