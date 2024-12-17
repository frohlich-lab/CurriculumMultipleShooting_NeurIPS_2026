import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Callable, Optional


class ResidualMLP(eqx.Module):
    """A Multi-Layer Perceptron (MLP) with skip connections."""
    mlp: eqx.nn.MLP
    skip_projection: Optional[eqx.nn.Linear] = None

    def __init__(
        self,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        use_bias: bool,
        use_final_bias: bool,
        activation: Callable,
        final_activation: Callable,
        key,
    ):
        """
        Initializes a ResidualMLP.

        Args:
            in_size: Input dimension size.
            out_size: Output dimension size.
            width_size: Hidden layer width.
            depth: Number of layers in the MLP.
            use_bias: Whether to use bias in the hidden layers.
            use_final_bias: Whether to use bias in the final layer.
            activation: Activation function to use between layers.
            final_activation: Activation function to use in the final layer.
            key: PRNG key for initializing weights.
        """
        if in_size != out_size:
            keys = jax.random.split(key, 2)
        else:
            keys = [key]
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            activation=activation,
            final_activation=final_activation,
            key=keys[0]
        )

        # If input size doesn't match output size, create a linear projection layer
        self.skip_projection = (
            eqx.nn.Linear(in_size, out_size, key=keys[1]) if in_size != out_size else None
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        if self.skip_projection is not None:
            residual = self.skip_projection(residual)
        return self.mlp.final_activation(self.mlp(x) + residual)


    def __getattr__(self, name: str):
        """
        Intercept attribute access and delegate to self.mlp if the attribute is not found in ResidualMLP.

        Args:
            name: Name of the attribute to access.

        Returns:
            The attribute from either self or self.mlp.
        """
        if hasattr(self.mlp, name):
            return getattr(self.mlp, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
