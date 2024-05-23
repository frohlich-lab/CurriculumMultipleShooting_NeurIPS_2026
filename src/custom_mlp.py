import equinox as eqx
import jax.random as jr

from typing import Any, Callable, List


class CustomDropoutMLP(eqx.Module):
    layers: List[eqx.nn.Linear]
    activation: Callable
    dropout_keys: List[Any]

    # TODO @GiacomoFabrini consider adding bias behaviour (use_bias)
    def __init__(
            self,
            in_size: int,
            out_size: int,
            width_size: int,
            depth: int,
            key: Any,
            activation: Callable,
            dropout_rate: float,
    ):
        # Initialise layers and prepare keys for layer initialisation
        self.layers = []
        lyr_key, dropout_key = jr.split(key)
        layer_keys = jr.split(lyr_key, num=depth+1)
        self.dropout_keys = jr.split(dropout_key, num=depth)

        # Define layer-wise architecture
        layer_sizes = [in_size] + [width_size] * depth + [out_size]
        for layer_num, (
                (fan_in, fan_out),
                key
        ) in enumerate(
            zip(
                zip(layer_sizes[:-1], layer_sizes[1:]),
                layer_keys
            )
        ):
            self.layers.append(
                eqx.nn.Linear(
                    in_features=fan_in,
                    out_features=fan_out,
                    # adds bias by default (use_bias=True) TODO @GiacomoFabrini - consider prescribing this properly
                    key=key,
                )
            )
            # Add dropout layer after every Linear layer except output
            if fan_out != out_size:
                self.layers.append(
                    eqx.nn.Dropout(p=dropout_rate)
                )
        self.activation = activation

    def __call__(self, x):
        a = x
        if len(self.layers) > 1:
            for i, layer in enumerate(self.layers[:-1]):
                if isinstance(layer, eqx.nn.Linear):
                    # Only apply activation after dense/linear layers
                    a = self.activation(layer(a))
                elif isinstance(layer, eqx.nn.Dropout):
                    a = layer(a, key=self.dropout_keys[i % 2 + 1])
        # If single layer (self.layers[0] == self.layers[-1]), fully linear behaviour (no non-linear activations).
        return self.layers[-1](a)
