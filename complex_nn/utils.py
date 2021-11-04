import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def initialize_cmplx_haiku_model(model, init_shape, rng_seed=42, **model_kwargs):

    def forward_pass(x, is_training):
        net = model(**model_kwargs)
        return net(x, is_training)

    key = jax.random.PRNGKey( rng_seed )

    dummy_input = jnp.zeros(init_shape, dtype='complex64')

    network = hk.transform_with_state(forward_pass)
    net_params, net_state = network.init( key, dummy_input, is_training=True )

    return network, net_params, net_state



def small_training_summary(history):

    fig, ax = plt.subplots(1, 3, figsize=(16,5))
    
    sns.lineplot(x=np.arange(len(history['train_loss'])), y=history['train_loss'], label='training_data', ax=ax[0])
    sns.lineplot(x=np.arange(len(history['val_loss'])), y=history['val_loss'], label='test', ax=ax[0])
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Linear loss')
    
    sns.lineplot(x=np.arange(len(history['train_loss'])), y=history['train_loss'], label='training_data', ax=ax[1])
    sns.lineplot(x=np.arange(len(history['val_loss'])), y=history['val_loss'], label='test', ax=ax[1])
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].set_yscale('log')
    ax[1].set_title('log loss')
    
    sns.lineplot(x=np.arange(len(history['train_acc'])), y=history['train_acc'], label='training_data', ax=ax[2])
    sns.lineplot(x=np.arange(len(history['val_acc'])), y=history['val_acc'], label='test_data', ax=ax[2])
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('Accuracy')
    ax[2].set_title('Model Accuracy')
