# -*- coding: utf-8 -*-
"""

HAIKU CLASSIFIER




-----------------------------------------
Example:





----------------------------------------


Last modified: 22/09/2021

Author:
     
     Mattia Pujatti,
     Physics of Data student,
     Internship at FBK-DSIP, Trento.
     mpujatti@fbk.eu   

----------------------------------------

"""

import jax
import haiku as hk
from jax import random, grad, jit, value_and_grad
import jax.numpy as jnp
from jax.experimental import optimizers
from functools import partial
import time
from tqdm.notebook import tqdm
import numpy as np

from sklearn import metrics
import pickle
import os

from typing import Optional, Any, Tuple, Callable

class Haiku_Classifier:
    """This class is formulated in order to simplify all the sequence of operations that one has to write
    in order to train a model written in Haiku. Initializing an instance of this class, fixed the 
    necessary hyperparameters, and simply calling the 'train' method on a dataset:

    The idea is giving in input to the training function a 'forward' function, in which one invokes the 
    '__call__' method, defined in an object of type 'hk.Module', with the necessary parameters.


    Attributes
    ----------
    forward_fn: function 
         function calling the network class (**check class description above)

    trained_parameters: dictionary
         Container of the trained network parameters (empty before calling the train function)

    rng_seq: hk.PRNGSequence
         Sequence of random jax keys. 
         With next(self.rng_seq) you have access to a new random key.

    with_state: bool
         Flag that enable the presence of an internal trainable state in the network (like Batchnorm). 
         When True, it transforms the model with 'hk.transform_with_state' and mantains a 'network state' 
         variable across the training. 

    network: transformed model
         Internal variable containing the model to train after the transformation of 'forward' into a pure
         function, according to haiku's abstraction technique (please refer to official documentation at
         https://dm-haiku.readthedocs.io/en/latest/notebooks/basics.html).
    
    Methods
    -------
    init(rng_seed, with_state)
         Initialize some attributes of the class.

    __categorical_accuracy(params, inputs, targets, rng_key, net_state)
         Compute the accuracy of the network's predictions for input data.
    
    __crossentropy_loss(params, inputs, targets, rng_key, net_state, is_training)
         Compute the categorical crossentropy loss of the network's predicitons over the input data.

    __update(step, params, opt_state, x, y, rng_key, net_state, is_training)
         Perform an update step of the parameters of the network.

    __initialize(forward_fn, init_dataloader)
        Implement the haiku transformation of the network's forward function and initialize 
        its parameters and state.
    
    train(n_epochs, forward_fn, optimizer, train_dataloader, test_dataloader, verbose)
         Setup and run the training process looping over both the train and validation sets in order
         to call the update function (when required), and collecting the values of accuracy and loss
         for each epoch.

    evaluate_dataset(dataloader)
         Once the network has been correctly trained, one can call this function and compute the values
         of accuracy and loss for the given dataset.

    compute_confusion_matrix(dataloader, normalize)
         Once the network has been correctly trained, one can call this function and compute the confusion
         matrix for the predictions of the given dataset.

    save_instance(name, path)
         Save an instance of the whole class in a pickle file.

    load_instance(path)
         Load an instance of the whole class and all its attributes from a pre-saved pickle file.
    
    """

    def __init__(self,
                 
                 rng_seed: Optional[int] = 42,
                 with_state: Optional[bool] = False
    ):
        """Initialize the attributes of the class.
        
        Args
        ----
        rng_seed: int, optional (default is 42)
             Initial seed to construct the PRNGSequence.
        with_state: bool, optional (default is False)
             Flag that enable/disable the tracking of the network state.
        """

        self.training_completed = False

        self.rng_seq = hk.PRNGSequence(random.PRNGKey(rng_seed))
        self.with_state = with_state

        
    @partial(jit, static_argnums=(0,))
    def __categorical_accuracy(self,
                               params: hk.Params,
                               inputs,
                               targets,
                               rng_key: Optional[jax.random.PRNGKey] =None,
                               net_state: Optional[hk.State] = None
    ) -> Any:
        """Compute the fraction of correctly classified samples by the network, for a given input batch.

        Args
        ----
        params: hk.Params
             Parameters of the network.
        inputs: array 
             Array of samples to give in input to the network.
        targets: array
             Array of one-hot encoded labels.
        rng_key: jax.random.PRNGKey, optional (default is None)
             PRNGKey necessary to the 'apply' method of the transformed network.
        net_state: hk.State, optional (defualt is None)
             Internal state of the network. Set 'None' if the network has no internal trainable state.

        Return
        ------
        categorical_accuracy: float
             Fraction of correctly classified samples by the network.
        """

        target_class = jnp.argmax(targets, axis=-1)
        if net_state is None:
            predictions = self.network.apply(params, rng_key, inputs, is_training=False)
        else:
            predictions, net_state = self.network.apply(params, net_state, rng_key, inputs, is_training=False)

        # Traditional accuracy is not defined for complex output
        predictions = jnp.absolute(predictions)
        predicted_class = jnp.argmax(predictions, axis=-1)

        return jnp.mean(predicted_class == target_class)
    

    @partial(jit, static_argnums=(0,6,))
    def __crossentropy_loss(self,
                            params: hk.Params,
                            inputs,
                            targets,
                            rng_key: Optional[jax.random.PRNGKey] =None,
                            net_state: Optional[hk.State] = None,
                            is_training: bool = False
    ) -> Tuple[Any, hk.State]:
        """Compute the categorical crossentropy loss between the samples given in input and the 
        corresponding network's predictions.

        Args
        ----
        params: hk.Module.parameters()
             Parameters of the network.
        inputs: array 
             Array of samples to give in input to the network.
        targets: array
             Array of one-hot encoded labels.
        rng_key: jax.random.PRNGKey, optional (default is None)
             PRNGKey necessary to the 'apply' method of the transformed network.
        net_state: , optional (defualt is None)
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
        
        if net_state is None:
            logits = self.network.apply(params, rng_key, inputs, is_training)
        else:
            logits, net_state = self.network.apply(params, net_state, rng_key, inputs, is_training)

        # Traditional cross-entropy is not defined for complex output
        logits = jnp.absolute(logits)

        # Add weigth regularization
        #l1_loss = sum(jnp.sum(jnp.abs(p)) for p in jax.tree_leaves(params))
        #l2_loss = jnp.sqrt(sum(jnp.vdot(x, x) for x in jax.tree_leaves(params))).real
        softmax_xent = -jnp.sum(targets * jax.nn.log_softmax(logits, axis=-1)) / len(targets)

        
        #total_loss = softmax_xent + 1e-4*l2_loss
        
        return softmax_xent, net_state




    @partial(jit, static_argnums=(0,8,))
    def __update(self,
                 step: int,
                 params: hk.Params,
                 opt_state: Any,
                 x,
                 y,
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

        (loss, net_state), grads = value_and_grad(self.__crossentropy_loss, has_aux=True)(params, x, y, rng_key, net_state, is_training)
        grads = jax.tree_multimap(jnp.conjugate, grads)
        #print(jax.tree_multimap(jnp.mean, grads))

        opt_state = self.opt_update(step, grads, opt_state)

        return self.get_params(opt_state), opt_state, loss, net_state



    def initialize(self,
                   model: hk.Module,
                   init_dataloader,
                   **model_kwargs
    ):
        """Implement the haiku transformation of the network's forward function and initialize 
        its parameters and state.

        Args
        ----
        forward_fn: function
             Forward function of the haiku.Module object defining the network.
        init_dataloader: pytorch DataLoader
             Batched dataloader necessary for the initialization of the parameters of the network.

        Return
        ------
        net_params: hk.Module.parameters
             Just initialized parameters of the network.
        net_state:
             Internal state of the network.
        """

        # Take a sample batch to initialize the parameters of the network
        init_batch = next(iter(init_dataloader))[0].numpy()

        # Construct a function to perform the 'forward propagation' step
        # this function will be transformed by haiku
        def forward_pass(x, is_training):
            net = model(**model_kwargs)
            return net(x, is_training)
        
        if self.with_state:
            self.network = hk.transform_with_state(forward_pass)
            net_params, net_state = self.network.init( next(self.rng_seq), init_batch, is_training=True )
        else:
            self.network = hk.transform(forward_pass)
            net_state = None
            net_params = self.network.init( next(self.rng_seq), init_batch, is_training=True )        


        return net_params, net_state


    
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
        net_params, net_state = self.initialize(model, train_dataloader, **model_kwargs)

        # Setup the optimizer
        # (refer to official documentation at https://jax.readthedocs.io/en/latest/jax.experimental.optimizers.html?highlight=optimizers)
        opt_init, self.opt_update, self.get_params = optimizer
        
        # Initialize the optimizer state
        opt_state = opt_init(net_params)

        training_history = {'train_loss': [],
                            'val_loss': [],
                            'train_acc': [],
                            'val_acc': [],
                        }
        step = 0
        for epoch in tqdm(range(n_epochs), desc='Training for several epochs', leave=False):

            start_time = time.time()
            log_train_loss, log_train_acc = [], []
            for batch in tqdm(train_dataloader, desc='Looping over the minibatches', leave=False):
                
                x_batch, y_batch = batch[0].numpy(), batch[1].numpy()

                net_params, opt_state, batch_loss, net_state = self.__update(
                    step, net_params, opt_state, x_batch, y_batch, next(self.rng_seq), net_state, is_training=True )

                batch_accuracy = self.__categorical_accuracy( net_params, x_batch, y_batch, next(self.rng_seq), net_state )

                log_train_loss.append(batch_loss)
                log_train_acc.append(batch_accuracy)

                step += 1

            training_history['train_loss'].append(np.mean(log_train_loss))
            training_history['train_acc'].append(np.mean(log_train_acc))

            log_val_loss, log_val_acc = [], []
            for batch in tqdm(test_dataloader, desc='Computing the validation loss', leave=False):

                x_batch, y_batch = batch[0].numpy(), batch[1].numpy()

                batch_loss, _ = self.__crossentropy_loss( net_params, x_batch, y_batch, next(self.rng_seq), net_state, is_training=False )                 
                batch_accuracy = self.__categorical_accuracy( net_params, x_batch, y_batch, next(self.rng_seq), net_state )

                log_val_loss.append(batch_loss)
                log_val_acc.append(batch_accuracy)


            training_history['val_loss'].append(np.mean(log_val_loss))
            training_history['val_acc'].append(np.mean(log_val_acc))

            if verbose:
                print("Training set loss {}".format(     training_history['train_loss'][-1]) )
                print("Test set loss {}".format(         training_history['val_loss'][-1]) )
                print("Training set accuracy {}".format( training_history['train_acc'][-1]) )
                print("Test set accuracy {}".format(     training_history['val_acc'][-1]) )

        self.training_completed = True
        self.trained_parameters = net_params
        self.trained_net_state = net_state
        self.training_history = training_history
        #self.opt_final_state = opt_state
        
        return training_history



    def evaluate_dataset(self,
                         dataloader
    ):
        """Once the network has been correctly trained, one can call this function and compute the values
         of accuracy and loss for the given dataset.

        Args
        ----
        dataloader: pytorch DataLoader
             Dataloader containing all the test samples.       
        
        """

        if self.training_completed:
            net_params = self.trained_parameters
            net_state = self.trained_net_state
        else:
            raise ValueError("Warning: the network has not been trained yet. Can't proceed with the evaluation.")


        log_val_loss, log_val_acc = [], []
        for batch in tqdm(dataloader, desc='Computing the accuracy / loss over the dataset.', unit='batches', leave=False):

            x_batch, y_batch = batch[0].numpy(), batch[1].numpy()

            batch_loss, _ = self.__crossentropy_loss( net_params, x_batch, y_batch, next(self.rng_seq), net_state, is_training=False )                 
            batch_accuracy = self.__categorical_accuracy( net_params, x_batch, y_batch, next(self.rng_seq), net_state )

            log_val_loss.append(batch_loss)
            log_val_acc.append(batch_accuracy)
            

        print('Final loss of the test set: {:.3f}'.format(np.mean(log_val_loss)))
        print('Final accuracy of the test set: {:.2f}%'.format(np.mean(log_val_acc)*100))

            
        

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

        if self.training_completed:
            net_params = self.trained_parameters
            net_state = self.trained_net_state
        else:
            raise ValueError("Warning: the network has not been trained yet. Can't proceed with the evaluation.")

        
        labels = []
        preds = []
        
        for batch in tqdm(dataloader, desc='Looping over the dataset.', unit='batches', leave=False):

            x_batch, y_batch = batch[0].numpy(), batch[1].numpy()
            labels.append(np.argmax(y_batch, axis=-1))
            
            if net_state is None:
                predictions = self.network.apply(net_params, next(self.rng_seq), x_batch, is_training=False)
            else:
                predictions, _ = self.network.apply(net_params, net_state, next(self.rng_seq), x_batch, is_training=False)

            preds.append(np.argmax(np.absolute(predictions), axis=-1))

        labels = np.concatenate(labels)
        preds = np.concatenate(preds)
                
        conf_mat = metrics.confusion_matrix(y_true = labels,
                                            y_pred = preds,
                                            normalize = normalize)

        return conf_mat



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

        if not self.training_completed:
            raise ValueError("Warning: the network has not been trained yet. Nothing to save.")

        if path is None:
            path = os.getcwd()

        file_path = os.path.join(path, name + ".pkl")
        model_path = os.path.join(path, name + "_model.pkl")

        # Unfortunately, saving haiku and JAX objects with pickle is not so easy, since several types are not
        # yet supported, especially haiku trasnformed functions (see https://github.com/deepmind/dm-haiku/issues/59)
        # So we are allowed to save only a few parameters
        backup_dict = {k: self.__dict__[k] for k in self.__dict__.keys() if k not in ['network', 'get_params', 'opt_update']}

        with open(file_path, "wb") as f:
            pickle.dump(backup_dict, f)

        with open(model_path, "wb") as f:
            pickle.dump(model, f)


    def load_instance(self,
                      class_path,
                      model_path,
                      init_dl,
                      **model_kwargs
    ):
        """Load an instance of the whole class and all its attributes.
        
        Args:
        ----
        path: str
             Directory of the pickle file

        """

        with open(class_path, "rb") as f:
            temp_haiku_classifier = pickle.load(f)

        self.__dict__ = temp_haiku_classifier

        # Network needs to be re-initialized
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            
        _, _ =  self.initialize(model, init_dl, **model_kwargs)



    def predict(self,
                samples
    ):
        """Wraps the 'apply' method of the network exploiting the trained parameters
        and state.

        Args:
        ----
        samples:
        
        Return
        ------
        """

        if not self.training_completed:
            raise ValueError("Warning: the network has not been trained yet. Can't proceed with the evaluation.")

        if self.with_state:
            predictions, _ = self.network.apply(self.trained_parameters, self.trained_net_state, next(self.rng_seq), samples, is_training=False)
        else:
            predictions = self.network.apply(self.trained_parameters, next(self.rng_seq), samples, is_training=False)

        return predictions
