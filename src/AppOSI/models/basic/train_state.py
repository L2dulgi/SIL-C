from jax import random
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

def create_train_state_basic(model, input_config, optimizer_config=None):
    """
    Create a TrainState for a model with a basic initialization procedure.
    """
    k1, k2 = random.split(random.PRNGKey(444), 2)
    r1, r2 = random.split(random.PRNGKey(777), 2)
    rngs = {'params': k1, 'dropout': r1}

    input_dict = {key: jnp.zeros(input_config[key]) for key in input_config.keys()}
    params = model.init(rngs=rngs, **input_dict)

    if optimizer_config is None:
        lr = 1e-5
        momentum = 0.9
        optimizer = optax.adam(lr, momentum)
    else:
        optimizer_cls = optimizer_config['optimizer_cls']
        optimizer = optimizer_cls(**optimizer_config['optimizer_kwargs'])

    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

def create_train_state_time_cond(model, input_config, optimizer_config):
    """
    Create a TrainState for a time-conditional model initialization.
    """
    rngs = {'params': random.PRNGKey(444), 'dropout': random.PRNGKey(44)}
    input_kwargs = {k: jnp.zeros(input_config[k]) for k in input_config.keys()}
    params = model.init(rngs, **input_kwargs)

    optimizer_cls = optimizer_config['optimizer_cls']
    optimizer = optimizer_cls(**optimizer_config['optimizer_kwargs'])

    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
