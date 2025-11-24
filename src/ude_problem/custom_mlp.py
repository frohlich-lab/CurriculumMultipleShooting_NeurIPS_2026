import equinox as eqx
import jax.random as jr

from typing import Any, Callable, Dict, List, Union


class CustomDropoutMLP(eqx.Module):
    layers: List[Union[eqx.nn.Linear, eqx.nn.Dropout]]
    activation: Callable
    final_activation: Callable
    use_bias: bool
    final_bias: bool
    dropout_keys: Any
    input_dropout_rate: float
    hidden_dropout_rate: float

    def __init__(
            self,
            in_size: int,
            out_size: int,
            width_size: int,
            depth: int,
            key: Any,
            use_bias: bool,
            use_final_bias: bool,
            activation: Callable,
            final_activation: Callable,
            dropout_rates: Dict[str, float],
    ):
        # Initialise layers and prepare keys for layer initialisation
        self.layers = []
        self.use_bias = use_bias
        self.final_bias = use_final_bias
        lyr_key, dropout_key = jr.split(key)
        layer_keys = jr.split(lyr_key, num=depth+1)
        self.dropout_keys = jr.split(dropout_key, num=depth)
        self.input_dropout_rate = dropout_rates["input"]
        self.hidden_dropout_rate = dropout_rates["hidden"]

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
            if fan_in == in_size:
                p = self.input_dropout_rate
            else:
                p = self.hidden_dropout_rate
            # Dropout layers on inputs and hidden layers, but not outputs
            self.layers.append(
                eqx.nn.Dropout(p=p)
            )
            self.layers.append(
                eqx.nn.Linear(
                    in_features=fan_in,
                    out_features=fan_out,
                    use_bias=self.use_bias if layer_num < depth else self.final_bias,
                    key=key,
                )
            )
        self.activation = activation
        self.final_activation = final_activation

    def __call__(self, x):
        a = x
        if len(self.layers) > 1:
            for i, layer in enumerate(self.layers[:-1]):
                if isinstance(layer, eqx.nn.Linear):
                    # Only apply activation after dense/linear layers
                    a = self.activation(layer(a))
                elif isinstance(layer, eqx.nn.Dropout):
                    a = layer(a, key=self.dropout_keys[i % 2 + 1])
        # by default the final activation is an identity (no non-linearity)
        return self.final_activation(self.layers[-1](a))
