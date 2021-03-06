import jax
import jax.numpy as jnp
from jax import random, grad, jit, value_and_grad
import haiku as hk
import numpy as np
from jax.experimental import optimizers
from functools import partial
from tqdm.auto import tqdm
from sklearn import metrics

from complex_nn.haiku_ml_utils import initialize_cmplx_haiku_model



class WDGRL:


    def __init__(self, rng_seed=42):

        self.rng_seq = hk.PRNGSequence(jax.random.PRNGKey(rng_seed))

        self.wdgrl_net_keys = ['feature_extractor', 'discriminator', 'critic']

        self.__networks   = {k: None for k in self.wdgrl_net_keys}
        self.__net_params = {k: None for k in self.wdgrl_net_keys}
        self.__net_states = {k: None for k in self.wdgrl_net_keys}

        self.__net_optimizers = {k: {k_opt: None for k_opt in ['opt_update', 'get_params', 'opt_state']} for k in self.wdgrl_net_keys}


        self.training_history = {'critic_loss'          : [],
                                 'features_loss'        : [],
                                 'discriminator_loss'   : [],
                                 'source_train_accuracy': [],
                                 'target_train_accuracy': [],
                                 'source_test_accuracy' : [],
                                 'target_test_accuracy' : []
                                 }

    def get_network(self, net_key):
        return self.__networks[net_key]
    
    def get_params(self, net_key):
        return self.__net_params[net_key]

    def update_params(self, net_key, params):
        self.__net_params[net_key] = params

    def get_net_state(self, net_key):
        return self.__net_states[net_key]

    def update_net_state(self, net_key, state):
        self.__net_states[net_key] = state

    def get_opt_state(self, net_key):
        return self.__net_optimizers[net_key]['opt_state']

    def update_opt_state(self, net_key, state):
        self.__net_optimizers[net_key]['opt_state'] = state

    def get_opt_updater(self, net_key):
        return self.__net_optimizers[net_key]['opt_update']

    def get_opt_getparams(self, net_key):
        return self.__net_optimizers[net_key]['get_params']


    
    def network_forward_pass(self, net_key, z, is_training=False):

        model = self.get_network( net_key )
        out, net_state = model.apply( self.get_params(net_key), self.get_net_state(net_key), next(self.rng_seq), z, is_training )

        if is_training:
            self.update_net_state(net_key, net_state)
 
        return out

    


    def init_network(self, model, input_shape, model_key, optimizer=None, **model_kwargs):

        model, init_net_params, init_net_state = initialize_cmplx_haiku_model(model, input_shape, **model_kwargs) 

        self.__networks[model_key] = model
        self.update_params(model_key, init_net_params)
        self.update_net_state(model_key, init_net_state)

        if optimizer is not None:
            opt_init, opt_update, get_params = optimizer
            opt_state = opt_init(init_net_params)

            self.__net_optimizers[model_key] = {'opt_update': opt_update, 'get_params': get_params, 'opt_state': opt_state}



    def init_wdgrl_networks(self, feature_extractor, fe_optimizer, discriminator, d_optimizer, critic, c_optimizer, input_shape):

        self.init_network(feature_extractor, input_shape, 'feature_extractor', fe_optimizer)

        dummy_input = jnp.zeros(input_shape, dtype='complex64')
        dummy_out_shape = self.network_forward_pass('feature_extractor', dummy_input).shape

        self.init_network(discriminator, dummy_out_shape, 'discriminator', d_optimizer)
        self.init_network(critic, dummy_out_shape, 'critic', c_optimizer)

                                  

    @partial(jit, static_argnums=(0,6,))
    def crossentropy_loss(self, disc_params, disc_state, features, targets, rng_key, is_training):

        logits, disc_state = self.get_network('discriminator').apply( disc_params, disc_state, rng_key, features, is_training)
        logits = jnp.absolute(logits)#**2

        softmax_xent = -jnp.sum(targets * jax.nn.log_softmax(logits, axis=-1)) / len(targets)

        return softmax_xent, disc_state

    @partial(jit, static_argnums=(0,))
    def update_discriminator(self, step, inputs, targets, disc_params, disc_state, disc_opt_state, rng_key):

        (loss, disc_state), grads = value_and_grad(self.crossentropy_loss, has_aux=True)(disc_params, disc_state, inputs, targets, rng_key, is_training=True)
        grads = jax.tree_multimap(jnp.conjugate, grads)

        new_opt_state = self.get_opt_updater('discriminator')(step, grads, disc_opt_state)
        new_params = self.get_opt_getparams('discriminator')(new_opt_state)

        return loss, new_opt_state, new_params, disc_state

        

    @partial(jit, static_argnums=(0,11,))
    def feature_extraction_loss(self, feat_params, feat_state, x_s, x_t, y_s, critic_params, critic_state, disc_params, disc_state, rng_key, is_training):

        LAMBDA = self.LAMBDA

        h_s, feat_state = self.get_network('feature_extractor').apply( feat_params, feat_state, rng_key, x_s, is_training)
        h_t, feat_state = self.get_network('feature_extractor').apply( feat_params, feat_state, rng_key, x_t, is_training)

        softmax_xent, _ = self.crossentropy_loss(disc_params, disc_state, h_s, y_s, rng_key, is_training=False)
        wasserstein_distance, _ = self.wasserstein_distance_loss(critic_params, critic_state, h_s, h_t, rng_key, is_training=False, GAMMA=0.)

        return softmax_xent + LAMBDA*wasserstein_distance, feat_state


    @partial(jit, static_argnums=(0,))
    def update_feature_extractor(self, step, x_s, x_t, y_s, feat_params, feat_state, feat_opt_state, critic_params, critic_state, disc_params, disc_state, rng_key):

        (loss, feat_state), grads = value_and_grad(self.feature_extraction_loss, has_aux=True)(feat_params, feat_state, x_s, x_t, y_s, critic_params, critic_state, disc_params, disc_state, rng_key, is_training=True)
        grads = jax.tree_multimap(jnp.conjugate, grads)

        new_opt_state = self.get_opt_updater('feature_extractor')(step, grads, feat_opt_state)
        new_params = self.get_opt_getparams('feature_extractor')(new_opt_state)

        return loss, new_opt_state, new_params, feat_state
        

    @partial(jit, static_argnums=(0,))
    def categorical_accuracy(self, inputs, targets, feat_params, feat_state, disc_params, disc_state, rng_key):

        target_class = jnp.argmax(targets, axis=-1)
        features, _ = self.get_network('feature_extractor').apply( feat_params, feat_state, rng_key, inputs, is_training=False)

        logits, _ = self.get_network('discriminator').apply(disc_params, disc_state, rng_key, features, is_training=False)
        logits = jnp.absolute(logits)
        predicted_class = jnp.argmax(logits, axis=-1)
        
        return jnp.mean(predicted_class == target_class)


    @partial(jit, static_argnums=(0,6,))
    def wasserstein_distance_loss(self, critic_params, critic_state, h_s, h_t, rng_key, is_training, GAMMA):

        critic_s, critic_state = self.get_network('critic').apply( critic_params, critic_state, rng_key, h_s, is_training)
        critic_t, critic_state = self.get_network('critic').apply( critic_params, critic_state, rng_key, h_t, is_training)

        wasserstein_distance = critic_s.mean() - critic_t.mean()
        wasserstein_distance = jnp.absolute(wasserstein_distance)
        critic_cost = wasserstein_distance - GAMMA*self.gradient_penalty(h_s, h_t, critic_params, critic_state, rng_key)

        return critic_cost, critic_state


    @partial(jit, static_argnums=(0,))
    def crit_norm(self, values, critic_params, critic_state, rng_key):
        values = jnp.expand_dims(values, 0)
        crits, _ = self.get_network('critic').apply( critic_params, critic_state, rng_key, values, is_training=False)
        return jnp.absolute(crits).squeeze()

    @partial(jit, static_argnums=(0,))
    def gradient_penalty(self, h_s, h_t, critic_params, critic_state, rng_key):

        alpha = jax.random.uniform(rng_key, shape=(h_s.shape[0], 1), dtype='float64')

        differences = h_t - h_s  
        interpolates = h_s + (alpha * differences) # (16, 392)
        interpolates = jnp.row_stack((interpolates, h_s, h_t))   # (48, 392)

        gradients = jax.vmap(jax.grad(self.crit_norm), in_axes=(0,None,None,None,), out_axes=0)(interpolates, critic_params, critic_state, rng_key)

        gradient_norm = jnp.linalg.norm(gradients, ord=2, axis=1)
        gradient_penalty = jnp.mean(((gradient_norm - 1)**2))

        return gradient_penalty


    @partial(jit, static_argnums=(0,))
    def update_critic(self, step, h_s, h_t, critic_params, critic_state, critic_opt_state, rng_key, GAMMA):

        (loss, critic_state), grads = value_and_grad(self.wasserstein_distance_loss, has_aux=True)(critic_params, critic_state, h_s, h_t, rng_key, is_training=True, GAMMA=GAMMA)
        grads = jax.tree_multimap(jnp.conjugate, grads)

        swap_sign = partial(jnp.multiply, x2=-1.)
        grads = jax.tree_multimap(swap_sign, grads)

        new_opt_state = self.get_opt_updater('critic')(step, grads, critic_opt_state)
        new_params = self.get_opt_getparams('critic')(new_opt_state)

        return loss, new_opt_state, new_params, critic_state

    
    def evaluate_classifier(self, *dataloaders):

        outs = []

        feat_params = self.get_params('feature_extractor')
        feat_state = self.get_net_state('feature_extractor')
        disc_params = self.get_params('discriminator')
        disc_state = self.get_net_state('discriminator')
        
        for dl in tqdm(dataloaders, desc='Evaluating the datasets.', leave=False):
            log_loss, log_acc = [], []
            for batch in dl:
                x_batch, y_batch = batch[0].numpy(), batch[1].numpy()

                log_acc.append( self.categorical_accuracy(x_batch, y_batch, feat_params, feat_state, disc_params, disc_state, next(self.rng_seq)) )

            outs.append( np.mean(log_acc) )
        return outs


    def compute_confusion_matrix(self, dataloader, normalize=None):

        labels = []
        preds = []

        for batch in dataloader:
            x_batch, y_batch = batch[0].numpy(), batch[1].numpy()

            labels.append(np.argmax(y_batch, axis=-1))

            predictions = self.network_forward_pass('discriminator', self.network_forward_pass('feature_extractor', x_batch))
            preds.append(np.argmax(np.absolute(predictions), axis=-1))

        labels = np.concatenate(labels)
        preds = np.concatenate(preds)
                
        conf_mat = metrics.confusion_matrix(y_true = labels, y_pred = preds, normalize=normalize)
        
        return conf_mat

        

    def run(self, n_epochs, n_iters, wdgrl_iters, iter_source, iter_target, source_train_loader_test, target_train_loader_test, source_test_loader, target_test_loader,
            GAMMA=100., LAMBDA=1., verbose=False):

        self.GAMMA = GAMMA
        self.LAMBDA = LAMBDA

        for epoch in tqdm(range(n_epochs), desc='Training for many epochs.', unit='epochs'):

            for it in tqdm(range(n_iters), desc='Looping over several minibatches.', unit='batches', leave=False):
                
                data_source, label_source = next(iter_source)
                data_target, label_target = next(iter_target)

                h_s = self.network_forward_pass('feature_extractor', data_source.numpy(), is_training=True)
                h_t = self.network_forward_pass('feature_extractor', data_target.numpy(), is_training=True)

                for wd_step in range(wdgrl_iters):
                    
                    critic_cost, critic_opt_state, critic_params, critic_state = self.update_critic(wd_step, h_s, h_t, self.get_params('critic'),
                                                                                                    self.get_net_state('critic'), self.get_opt_state('critic'), next(self.rng_seq), GAMMA=self.GAMMA)
                    self.update_net_state('critic', critic_state)
                    self.update_params('critic', critic_params)
                    self.update_opt_state('critic', critic_opt_state)
                    

                discriminator_loss, disc_opt_state, disc_params, disc_state = self.update_discriminator(it, h_s, label_source.numpy(), self.get_params('discriminator'),
                                                                                                    self.get_net_state('discriminator'), self.get_opt_state('discriminator'), next(self.rng_seq))
                self.update_net_state('discriminator', disc_state)
                self.update_params('discriminator', disc_params)
                self.update_opt_state('discriminator', disc_opt_state)

                feature_loss, feat_opt_state, feat_params, feat_state = self.update_feature_extractor(it, data_source.numpy(), data_target.numpy(), label_source.numpy(),
                                                                                                      self.get_params('feature_extractor'), self.get_net_state('feature_extractor'),
                                                                                                      self.get_opt_state('feature_extractor'),
                                                                                                      self.get_params('critic'), self.get_net_state('critic'),
                                                                                                      self.get_params('discriminator'), self.get_net_state('discriminator'), next(self.rng_seq))
                self.update_net_state('feature_extractor', feat_state)
                self.update_params('feature_extractor', feat_params)
                self.update_opt_state('feature_extractor', feat_opt_state)
                                                     
            source_train_acc, target_train_acc, source_test_acc, target_test_acc = self.evaluate_classifier(source_train_loader_test, target_train_loader_test, source_test_loader, target_test_loader)

            if verbose:
                print(f'Epoch: {epoch+1},\t Critic loss: {critic_cost},\t Feat extractor loss: {feature_loss},\t Discriminator loss: {discriminator_loss}')
                print(f'\t Source train accuracy: {source_train_acc}, \t Source test accuracy: {source_test_acc}, \t Target train accuracy: {target_train_acc}, \t Target test accuracy: {target_test_acc}')    
    
            self.training_history['critic_loss'].append(critic_cost)
            self.training_history['features_loss'].append(feature_loss)
            self.training_history['discriminator_loss'].append(discriminator_loss)
            self.training_history['source_train_accuracy'].append(source_train_acc)
            self.training_history['target_train_accuracy'].append(target_train_acc)
            self.training_history['source_test_accuracy'].append(source_test_acc)
            self.training_history['target_test_accuracy'].append(target_test_acc)

        return self.training_history
 


