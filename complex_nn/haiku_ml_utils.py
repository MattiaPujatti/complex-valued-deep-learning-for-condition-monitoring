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


def evaluate_model(network,
                   net_params,
                   net_state,
                   metric,
                   dataloader
    ) -> Any:
    """Calling this function one can compute the `metric` estimate over the input
    dataset, provided as a Pytorch dataloader.
    
    Args
    ----
    dataloader: pytorch DataLoader
        Dataloader containing all the test samples.       
    
    """

    log_metric = []
    for batch in tqdm(dataloader, desc='Looping over the dataset.', unit='batches', leave=False):

        x_batch, y_batch = batch[0].numpy(), batch[1].numpy()

        batch_metric, _ = metric( model, net_params, x_batch, y_batch, next(self.rng_seq), net_state, is_training=False )                 
        log_metric.append( batch_metric )

    return np.mean(log_metric)


@partial(jit, static_argnums=(0,8,))
def update(
    params: hk.Params,
    x: jnp.array,
    y: jnp.array,
    metric: Callable,
    opt_state: Any,
    opt_update: Callable,
    get_params: Callable,
    rng_key: Optional[jax.random.PRNGKey] =None,
    net_state: Optional[hk.State] = None,
    is_training: bool = False,
    step: int = 0,
) -> Tuple[hk.Params, Any, float, hk.State]:
    """Given a minibatch of samples, it compute the loss and the parameters updates of the network.
    Then, since jax.grad calculate the complex gradient df/dz and not the conjugate (as needed by 
    the complex backpropagation), an additional step performs this operation, before applying the 
    updates just computed.

    Args
    ----
    
    params: hk.Module.parameters()
        Parameters of the network.
    opt_state: jax pytree
        Object representing the actual optimizer state.
    x: array 
        Array of samples to give in input to the network.
    y: array
        Array of one-hot encoded labels.
    rng_key: jax.random.PRNGKey, optional (default is None)
        PRNGKey necessary to the 'apply' method of the transformed network.
    net_state: hk.State, optional (default is None)
        Internal state of the network. Set 'None' if the network has no internal trainable state.
    is_training: bool, optional (default is False)
        Flags that alert the network if it is called in training or evaluation mode. Useful in presence
        of dropout or batchnormalization layers.
    step: int, optional (default is 0)
        Index of the update step.

    Return
    ------
    new_params: hk.Module.parameters
        New estimates of network's parameters.
    opt_state: jax pytree
        Optimizer state after the update.
    loss: float
        Loss estimate for the given minibatch.
    net_state:
        Internal state of the network.
    """
    
    (loss, net_state), grads = value_and_grad(self.__crossentropy_loss, has_aux=True)(params, x, y, rng_key, net_state, is_training)
    grads = jax.tree_multimap(jnp.conjugate, grads)
    #print(jax.tree_multimap(jnp.mean, grads))

    opt_state = self.opt_update(step, grads, opt_state)
    
    return self.get_params(opt_state), opt_state, loss, net_state
