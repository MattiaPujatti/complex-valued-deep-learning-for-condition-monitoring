import haiku as hk
import jax
from jax import jit
import jax.numpy as jnp
from functools import partial
from typing import Optional, Any


@partial(jit, static_argnums=(0,))
def categorical_accuracy(network,
                         params: hk.Params,
                         inputs: jnp.array,
                         targets: jnp.array,
                         rng_key: Optional[jax.random.PRNGKey] =None,
                         net_state: Optional[hk.State] = None
    ) -> Any:
    """Compute the fraction of correctly classified samples by the network, for a given input batch.
    
    Args
    ----
    params: hk.Params
        Parameters of the network.
    inputs: array 
        Array of samples to give in input to the network. Must have shape [N, ...]
    targets: array
        Array of categorical (shape [N,1]) or one-hot encoded (shape [N,n_classes]) labels.
    rng_key: jax.random.PRNGKey, optional (default is None)
        PRNGKey necessary to the 'apply' method of the transformed network.
    net_state: hk.State, optional (default is None)
        Internal state of the network. Set 'None' if the network has no internal trainable state.

    Return
    ------
    categorical_accuracy: float
        Fraction of correctly classified samples by the network.
    """
    # Check shape correctness for the targets
    assert len(targets.shape) == 2

    # Verify to have the same number of samples among inputs and targets
    assert inputs.shape[0] == targets.shape[0]

    # Check if samples are one-hot encoded or categorical
    if targets.shape[-1] == 1:
        target_class = targets.flatten()
    else:
        target_class = jnp.argmax(targets, axis=-1)
        
    predictions, net_state = network.apply(params, net_state, rng_key, inputs, is_training=False)
        
    # Traditional accuracy is not defined for complex output
    predictions = jnp.absolute(predictions)
    predicted_class = jnp.argmax(predictions, axis=-1)
    
    return jnp.mean(predicted_class == target_class)
