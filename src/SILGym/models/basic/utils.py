import jax 
import jax.numpy as jnp
import numpy as np
from inspect import isfunction
from einops import repeat

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def update_rngs(rngs):
    """
    Update each RNG key in a dict for the next step by splitting it.
    """
    for k in rngs.keys():
        rngs[k], _ = jax.random.split(rngs[k])
    return rngs

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :param repeat_only: if True, just repeats the raw timesteps instead of sinusoidal embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = jnp.exp(
            -jnp.log(max_period) * jnp.arange(start=0, stop=half, dtype=np.float32) / half
        )
        args = timesteps[:, jnp.newaxis] * freqs[jnp.newaxis]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding