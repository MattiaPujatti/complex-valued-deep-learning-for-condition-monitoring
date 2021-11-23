import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np



def initialize_cmplx_haiku_model(model, init_shape, data_type='complex64', rng_seed=42, **model_kwargs):

    def forward_pass(x, is_training):
        net = model(**model_kwargs)
        return net(x, is_training)

    key = jax.random.PRNGKey( rng_seed )

    dummy_input = jnp.zeros(init_shape, dtype=data_type)

    network = hk.transform_with_state(forward_pass)
    net_params, net_state = network.init( key, dummy_input, is_training=True )

    return network, net_params, net_state


def initialize_CAM_model(model, init_shape, data_type='complex64', rng_seed=42, **model_kwargs):

    def forward_pass(x, is_training, return_blobs=False):
        net = model(**model_kwargs)
        return net(x, is_training)

    key = jax.random.PRNGKey( rng_seed )

    dummy_input = jnp.zeros(init_shape, dtype=data_type)

    network = hk.transform_with_state(forward_pass)
    net_params, net_state = network.init( key, dummy_input, is_training=True, return_blobs=False )

    return network, net_params, net_state



def haiku_check_model_parameters(model, init_shape, data_type, verbose=True, **model_kwargs):

    _, model_params, _ = initialize_cmplx_haiku_model(model, init_shape, data_type, **model_kwargs)
    
    if verbose:
        print(dict(jax.tree_map(lambda x: x.shape, model_params)))

    n_params = sum(jax.tree_map(lambda x: np.prod(x.shape), jax.tree_flatten(model_params)[0]))
    params_dtype = jax.tree_flatten(model_params)[0][0].dtype

    print(f"Total number of parameters in the model: {n_params}")
    print(f"Parameters' dtype: {params_dtype}")
