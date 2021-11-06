import haiku as hk
import jax
from jax import jit
import jax.numpy as jnp
from functools import partial
from typing import Optional, Any, Tuple


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
    network: haiku model after hk.transform() and model.init().
        Transfomed forward function of an Haiku model (such that we can call `network.apply(..)`).
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




@partial(jit, static_argnums=(1,6,))
def crossentropy_loss(params: hk.Params,
                      network,
                      inputs: jnp.array,
                      targets: jnp.array,
                      rng_key: Optional[jax.random.PRNGKey] =None,
                      net_state: Optional[hk.State] = None,
                      is_training: bool = False
    ) -> Tuple[Any, hk.State]:
    """Compute the categorical crossentropy loss between the samples given in input and the 
    corresponding network's predictions.
    
    Args
    ----
    network: haiku model after hk.transform() and model.init().
        Transfomed forward function of an Haiku model (such that we can call `network.apply(..)`).
    params: hk.Module.parameters()
        Parameters of the network.
    inputs: array 
        Array of samples to give in input to the network. Must have shape [N,...].
    targets: array
        Array of one-hot encoded labels. Must have shape [N, n_classes].
    rng_key: jax.random.PRNGKey, optional (default is None)
        PRNGKey necessary to the 'apply' method of the transformed network.
    net_state: , optional (default is None)
        Internal state of the network. Set 'None' if the network has no internal trainable state.
    is_training: bool, optional (default is False)
        Flags that alert the network if it is called in training or evaluation mode. Useful in presence
        of dropout or batchnormalization layers.

    Return
    ------
    softmax_xent: float
        Estimate of the crossentropy loss for the input batch.
    net_state:
        Actual internal state of the network.
    """

    logits, net_state = network.apply(params, net_state, rng_key, inputs, is_training)

    # Traditional cross-entropy is not defined for complex output
    logits = jnp.absolute(logits)
    
    # Check shape correctness (at this point they shoud both have shape [N, n_classes]
    assert logits.shape == targets.shape 
        
    # Add weigth regularization
    #l1_loss = sum(jnp.sum(jnp.abs(p)) for p in jax.tree_leaves(params))
    #l2_loss = jnp.sqrt(sum(jnp.vdot(x, x) for x in jax.tree_leaves(params))).real
    softmax_xent = -jnp.sum(targets * jax.nn.log_softmax(logits, axis=-1)) / len(targets)
    
    #total_loss = softmax_xent + 1e-4*l2_loss
        
    return softmax_xent, net_state
