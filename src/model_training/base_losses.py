import jax.numpy as jnp

from jaxtyping import Array

def get_mse(pred: Array, data: Array):
    return jnp.mean((pred - data) ** 2)


def get_masked_mse(pred: Array, data: Array, mask: Array):
    # Masked mean square error - mean is computed only over non-zero terms
    return jnp.sum(jnp.expand_dims(mask, axis=-1) * (pred - data) ** 2) / (jnp.sum(mask) * pred.shape[-1])
