import jax
import jax.numpy as jnp
import haiku as hk
from tqdm.notebook import tqdm
from typing import Optional, Tuple, Any


def initialize_cmplx_haiku_model(model, init_shape, rng_seed=42, **model_kwargs):

    def forward_pass(x, is_training):
        net = model(**model_kwargs)
        return net(x, is_training)

    key = jax.random.PRNGKey( rng_seed )

    dummy_input = jnp.zeros(init_shape, dtype='complex64')

    network = hk.transform_with_state(forward_pass)
    net_params, net_state = network.init( key, dummy_input, is_training=True )

    return network, net_params, net_state


