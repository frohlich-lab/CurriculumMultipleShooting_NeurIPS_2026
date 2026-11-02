"""Small convolutional network for grid-based (PDE) UDE/NODE examples.

The repo otherwise builds only dense MLPs (see ``build_mlp``). The reaction-diffusion example
(APHYNITY) needs a translation-equivariant network over the grid, so this module provides a compact
ConvNet with periodic (circular) padding, matching APHYNITY's 3-layer 3x3 ConvNet.

``ConvNet`` deliberately exposes the same metadata attributes as ``eqx.nn.MLP``
(``in_size``, ``out_size``, ``width_size``, ``depth``, ``use_bias``, ``use_final_bias``,
``activation``, ``final_activation``) so that ``UDEProblem.get_hyperparams`` / ``save`` keep working
unchanged. (Re-loading a serialised ConvNet via ``UDEProblem.load`` is not supported - it rebuilds an
MLP skeleton - but that path is only used by post-hoc analysis tools, not the training pipeline.)

NOTE on APHYNITY parity: for the reaction-diffusion example we keep this ConvNet as a biased,
circular-padded, BatchNorm-free 3-layer 3x3 net (hidden=16). This intentionally differs from
APHYNITY's ``ConvNetEstimator`` (BatchNorm2d after conv 1-2, ``bias=False`` on those convs, zero
padding on the NN convs). BatchNorm is omitted on purpose: inside an adaptive diffrax ODE RHS it
would renormalise the vector field on every evaluation and can destabilise the solve. See the
"Divergences from APHYNITY" note in
``src/model_examples/reaction_diffusion/setup_reaction_diffusion_model.py``.
"""
from typing import Callable, List

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import Array


def _identity(x):
    return x


class ConvNet(eqx.Module):
    layers: List[eqx.nn.Conv2d]
    activation: Callable = eqx.field(static=True)
    final_activation: Callable = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)
    # MLP-compatible metadata (consumed by UDEProblem.get_hyperparams)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    width_size: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    use_final_bias: bool = eqx.field(static=True)
    # Padding must match the PDE's boundary conditions, otherwise the network is asked to learn a
    # reaction term under a different boundary assumption than the physics it is coupled to:
    #   "wrap" -> periodic,  "edge" -> Neumann / zero-flux (replicated ghost cells).
    padding_mode: str = eqx.field(static=True, default="wrap")

    def __call__(self, x: Array) -> Array:
        """Forward pass. ``x`` has shape ``(C, H, W)``; output has shape ``(C_out, H, W)``."""
        pad = self.kernel_size // 2
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            # boundary-consistent padding, then a padding-free conv
            x = jnp.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode=self.padding_mode)
            x = layer(x)
            x = self.activation(x) if i < n - 1 else self.final_activation(x)
        return x


def build_conv_net(
    in_channels: int,
    out_channels: int,
    hidden_channels: int,
    n_layers: int,
    key,
    use_bias: bool = True,
    use_final_bias: bool = True,
    activation: Callable = jnn.relu,
    final_activation: Callable = _identity,
    kernel_size: int = 3,
    padding_mode: str = "wrap",
    init_scale: float = 1.0,
) -> ConvNet:
    """Build an ``n_layers``-deep ConvNet mapping ``in_channels`` -> ``out_channels`` feature maps.

    Channel sizes are ``[in, hidden, ..., hidden, out]``. Padding is applied manually in the
    forward pass, so each ``Conv2d`` uses ``padding=0``. ``padding_mode`` must match the PDE's
    boundary conditions: "wrap" for periodic, "edge" for Neumann.

    ``init_scale`` multiplies the FINAL layer's weight and bias only, so it scales the network's
    output magnitude directly while leaving signal propagation through the earlier layers at the
    standard initialisation. (Scaling every layer would compound: with ``n_layers`` layers the
    output would shrink by roughly ``init_scale ** n_layers``, so 0.1 on a 3-layer net means a
    ~1000x smaller output rather than 10x.)

    A smaller output means the untrained network contributes a smaller vector field, which makes
    the initial ODE cheaper to integrate. This matters most for the NODE variant, where the network
    IS the whole right-hand side; the UDE is dominated by its known diffusion term either way.
    """
    keys = jax.random.split(key, n_layers)
    channel_sizes = [in_channels] + [hidden_channels] * (n_layers - 1) + [out_channels]
    layers = [
        eqx.nn.Conv2d(
            in_channels=channel_sizes[i],
            out_channels=channel_sizes[i + 1],
            kernel_size=kernel_size,
            padding=0,
            use_bias=(use_bias if i < n_layers - 1 else use_final_bias),
            key=keys[i],
        )
        for i in range(n_layers)
    ]
    if init_scale != 1.0:
        last = layers[-1]
        layers[-1] = eqx.tree_at(
            lambda l: [l.weight] + ([l.bias] if l.bias is not None else []),
            last,
            replace=[last.weight * init_scale]
                    + ([last.bias * init_scale] if last.bias is not None else []),
        )
    return ConvNet(
        layers=layers,
        activation=activation,
        final_activation=final_activation,
        kernel_size=kernel_size,
        in_size=in_channels,
        out_size=out_channels,
        width_size=hidden_channels,
        depth=n_layers,
        use_bias=use_bias,
        use_final_bias=use_final_bias,
        padding_mode=padding_mode,
    )
