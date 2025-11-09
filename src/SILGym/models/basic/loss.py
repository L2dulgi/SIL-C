import jax.numpy as jnp

def mse(a, b):
    """
    Simple mean squared error between a and b.
    """
    def squared_error(a_, b_):
        return jnp.mean(jnp.square(a_ - b_), axis=-1)
    return squared_error(a, b)