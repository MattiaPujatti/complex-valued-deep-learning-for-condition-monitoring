import jax
import haiku as hk
from jax import random, grad, jit, value_and_grad
import jax.numpy as jnp
from jax.experimental import optimizers
from functools import partial
import time
from tqdm.notebook import tqdm
import numpy as np

from sklearn.metrics import confusion_matrix
import pickle
import os

from complex_nn.metrics import crossentropy_loss, categorical_accuracy
from complex_nn.haiku_ml_utils import initialize_cmplx_haiku_model

from typing import Optional, Any, Tuple, Callable

class Haiku_Classifier:
    """This class is formulated in order to simplify all the sequence of operations that one has to write
    in order to train a model written in Haiku. Initializing an instance of this class, fixed the 
    necessary hyperparameters, and simply calling the 'train' method on a dataset:

    The idea is giving in input to the training function a 'forward' function, in which one invokes the 
    '__call__' method, defined in an object of type 'hk.Module', with the necessary parameters.
    """

    def __init__(self,
                 rng_seed: Optional[int] = 42,
    ):
        """Initialize the attributes of the class.
        
        Args
        ----
        rng_seed: int, optional (default is 42)
             Initial seed to construct the PRNGSequence.
        """
        self.rng_seq = hk.PRNGSequence(random.PRNGKey(rng_seed))

        self.__network    = None
        self.__net_params = None
        self.__net_state  = None
        self.__opt_state  = None
        self.training_history = { 'train_loss': [], 'val_loss'  : [],
                                  'train_acc' : [], 'val_acc'   : [] }


    def get_network(self):
        return self.__network

    def get_net_params(self):
        return self.__net_params

    def update_net_params(self, params):
        self.__net_params = params

    def get_net_state(self):
        return self.__net_state

    def update_net_state(self, state):
        self.__net_state = state

    def get_opt_state(self):
        return self.__opt_state

    def update_opt_state(self, state):
        self.__opt_state = state


    def forward_pass(self, z):

        model = self.get_network()
        out, _ = model.apply( self.get_net_params(), self.get_net_state(), next(self.rng_seq), z, is_training=False) 

        return out


    @partial(jit, static_argnums=(0,8,))
    def __update(self,
                 step: int,
                 params: hk.Params,
                 opt_state: Any,
                 x: jnp.array,
                 y: jnp.array,
                 rng_key: Optional[jax.random.PRNGKey] =None,
                 net_state: Optional[hk.State] = None,
                 is_training: bool = False
    ) -> Tuple[hk.Params, Any, float, hk.State]:
        """Given a minibatch of samples, it compute the loss and the parameters updates of the network.
        Then, since jax.grad calculate the complex gradient df/dz and not the conjugate (as needed by 
        the complex backpropagation), an additional step performs this operation, before applying the 
        updates just computed.

        Args
        ----
        step: int
             Index of the update step.
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
        net_state: , optional (default is None)
             Internal state of the network. Set 'None' if the network has no internal trainable state.
        is_training: bool, optional (default is False)
             Flags that alert the network if it is called in training or evaluation mode. Useful in presence
             of dropout or batchnormalization layers.

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

        model = self.get_network()
        (loss, net_state), grads = value_and_grad(crossentropy_loss, has_aux=True)(params, model, x, y, rng_key, net_state, is_training)
        grads = jax.tree_multimap(jnp.conjugate, grads)
        #print(jax.tree_multimap(jnp.mean, grads))

        opt_state = self.opt_update(step, grads, opt_state)

        return self.get_params(opt_state), opt_state, loss, net_state



    
    def train(self,
              n_epochs: int,
              model: hk.Module,
              optimizer: optimizers.Optimizer,  
              train_dataloader,
              test_dataloader,
              verbose: bool = False,
              **model_kwargs
    ) -> Any:
        """Setup and run the training process looping over both the train and validation sets in order
         to call the update function (when required), and to collect the values of accuracy and loss
         for each epoch.
    
        Args
        ----
        n_epochs: int
             Number of epochs of the training loop.
        forward_fn: function
             Forward function of the haiku.Module object defining the network.
        optimizer: haiku.experimental.optimizer
             One of the optimizers proposed by haiku.
        train_dataloader: pytorch DataLoader
             Dataloader containing all the training samples
        test_dataloader: pytorch DataLoader
             Dataloader containing all the validation samples
        verbose: bool, optional (default is False)
             Verbosity of the output

        Return
        ------
        training_history: dict
             Dictionary containing the train/validation losses and accuracies for each epoch.
        """

        # Initialize the network
        batch_shape = next(iter(train_dataloader))[0].shape
        model, init_net_params, init_net_state = initialize_cmplx_haiku_model(model, batch_shape, **model_kwargs)

        self.__network = model
        self.update_net_params(init_net_params)
        self.update_net_state(init_net_state)

        # Setup the optimizer
        # (refer to official documentation at https://jax.readthedocs.io/en/latest/jax.experimental.optimizers.html?highlight=optimizers)
        opt_init, self.opt_update, self.get_params = optimizer
        
        # Initialize the optimizer state
        self.update_opt_state( opt_init(self.get_net_params()) )

        step = 0
        for epoch in tqdm(range(n_epochs), desc='Training for several epochs', leave=False):

            start_time = time.time()
            log_train_loss, log_train_acc = [], []
            for batch in tqdm(train_dataloader, desc='Looping over the minibatches', leave=False):
                
                x_batch, y_batch = batch[0].numpy(), batch[1].numpy()

                net_params, opt_state, batch_loss, net_state = self.__update(
                    step, self.get_net_params(), self.get_opt_state(), x_batch, y_batch, next(self.rng_seq), self.get_net_state(), is_training=True )

                # Update the corresponding global variables
                self.update_net_params( net_params )
                self.update_net_state( net_state )
                self.update_opt_state( opt_state )

                batch_accuracy = categorical_accuracy( model, self.get_net_params(), x_batch, y_batch, next(self.rng_seq), self.get_net_state() )
                
                log_train_loss.append(batch_loss)
                log_train_acc.append(batch_accuracy)

                step += 1

            self.training_history['train_loss'].append(np.mean(log_train_loss))
            self.training_history['train_acc'].append(np.mean(log_train_acc))

            log_val_loss, log_val_acc = [], []
            for batch in tqdm(test_dataloader, desc='Computing the validation metrics', leave=False):

                x_batch, y_batch = batch[0].numpy(), batch[1].numpy()

                batch_loss, _ = crossentropy_loss( self.get_net_params(), model, x_batch, y_batch, next(self.rng_seq), self.get_net_state(), is_training=False )                 
                batch_accuracy = categorical_accuracy( model, self.get_net_params(), x_batch, y_batch, next(self.rng_seq), self.get_net_state() )

                log_val_loss.append(batch_loss)
                log_val_acc.append(batch_accuracy)

            self.training_history['val_loss'].append(np.mean(log_val_loss))
            self.training_history['val_acc'].append(np.mean(log_val_acc))

            if verbose:
                print("Training set loss {}".format(     self.training_history['train_loss'][-1]) )
                print("Test set loss {}".format(         self.training_history['val_loss'][-1]) )
                print("Training set accuracy {}".format( self.training_history['train_acc'][-1]) )
                print("Test set accuracy {}".format(     self.training_history['val_acc'][-1]) )
        
        return self.training_history



    def evaluate_dataset(self,
                         *dataloaders
    ):
        """Once the network has been correctly trained, one can call this function and compute the values
         of accuracy and loss for the given dataset.

        Args
        ----
        dataloader: pytorch DataLoader
             Dataloader containing all the test samples.       
        
        """
        
        model = self.get_network()

        for dl in tqdm(dataloaders, desc='Evaluating the datasets.', leave=False):

            log_val_loss, log_val_acc = [], []
            for batch in tqdm(dl, desc='Computing the accuracy / loss over the dataset.', unit='batches', leave=False):

                x_batch, y_batch = batch[0].numpy(), batch[1].numpy()

                batch_loss, _ = crossentropy_loss( self.get_net_params(), model, x_batch, y_batch, next(self.rng_seq), self.get_net_state(), is_training=False )                 
                batch_accuracy = categorical_accuracy( model, self.get_net_params(), x_batch, y_batch, next(self.rng_seq), self.get_net_state() )

                log_val_loss.append(batch_loss)
                log_val_acc.append(batch_accuracy)
            

            print('Average loss of the set: {:.3f}'.format(np.mean(log_val_loss)))
            print('Average accuracy over the set: {:.2f}%'.format(np.mean(log_val_acc)*100))
            
        

    def compute_confusion_matrix(self,
                                 dataloader,
                                 normalize = 'true'
    ):
        """Compute the confusion matrix for the predictions over the input dataloader.

        Args
        ----
        dataloader: pytorch DataLoader
             Dataloader containing all the test batches.
        normalize: str or None, optional, default is 'true'
             Corresponds to the normalize parameter in sklearn.metrics.confusion_matrix
        Return
        ------
        confusion_matrix: array
             Confusion matrix of the network predictions.
        """
        
        labels = []
        preds = []
        
        for batch in tqdm(dataloader, desc='Looping over the dataset.', unit='batches', leave=False):

            x_batch, y_batch = batch[0].numpy(), batch[1].numpy()
            labels.append(np.argmax(y_batch, axis=-1))

            predictions = self.forward_pass( x_batch )
            preds.append(np.argmax(np.absolute(predictions), axis=-1))

        labels = np.concatenate(labels)
        preds = np.concatenate(preds)
                
        return confusion_matrix(y_true=labels, y_pred=preds, normalize=normalize)




    def save_instance(self,
                      name,
                      model: hk.Module,
                      path = None
    ):
        """Save an instance of the whole class and all its attributes (like the trained parameters, state,...).
        
        Args:
        ----
        name: str
             Name of the pickle file to be saved.
        path: str
             Directory in which the pickle file will be saved.

        """

        if path is None:
            path = os.getcwd()

        file_path = os.path.join(path, name + ".pkl")
        model_path = os.path.join(path, name + "_model.pkl")

        # Unfortunately, saving haiku and JAX objects with pickle is not so easy, since several types are not
        # yet supported, especially haiku trasnformed functions (see https://github.com/deepmind/dm-haiku/issues/59)
        # So we are allowed to save only a few parameters (mainly the network's parameters and states, the optimizer
        # state and the training history)
        backup_dict = {'net_params': self.get_net_params(),
                       'net_state': self.get_net_state(),
                       'opt_state': self.get_opt_state(),
                       'training_history': self.training_history}

        with open(file_path, "wb") as f:
            pickle.dump(backup_dict, f)

        with open(model_path, "wb") as f:
            pickle.dump(model, f)


    def load_instance(self,
                      class_path,
                      model_path,
                      init_dataloader,
                      **model_kwargs
    ):
        """Load an instance of the whole class and all its attributes.
        
        Args:
        ----
        path: str
             Directory of the pickle file

        """

        with open(class_path, "rb") as f:
            classifier_state = pickle.load(f)

        self.update_net_params( classifier_state['net_params'] )
        self.update_net_state( classifier_state['net_state'] )
        self.update_opt_state( classifier_state['opt_state'] )     
        self.training_history = classifier_state['training_history']

        # Network needs to be re-initialized
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Re-initialize the network
        batch_shape = next(iter(init_dataloader))[0].shape
        network, _, _ = initialize_cmplx_haiku_model(model, batch_shape, **model_kwargs)
        self.__network = network
            



