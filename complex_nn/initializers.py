""" Reframe haiku initializers to allow complex initialization of the network's weights. """

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Sequence


class CmplxRndUniform(hk.initializers.Initializer):
    """Initializes by sampling from a uniform distribution."""
    
    def __init__(self, minval=0, maxval=1.):
        """Constructs a :class:`RandomUniform` initializer.
        
        Args:
          minval: The lower limit of the uniform distribution.
          maxval: The upper limit of the uniform distribution.
        """
        self.minval = minval
        self.maxval = maxval
        
    def __call__(self, shape: Sequence[int], dtype: Any) -> jnp.ndarray:
        real_part = jax.random.uniform(hk.next_rng_key(), shape, dtype='float64', minval=self.minval, maxval=self.maxval)
        imag_part = jax.random.uniform(hk.next_rng_key(), shape, dtype='float64', minval=self.minval, maxval=self.maxval)
        return jax.lax.complex(real_part, imag_part)
    
class CmplxTruncatedNormal(hk.initializers.Initializer):
    """Initializes by sampling from a truncated normal distribution."""
    
    def __init__(self, mean=0., stddev=1., lower=-2., upper=2.):
        """Constructs a :class:`TruncatedNormal` initializer.
        
        Args:
           stddev: The standard deviation parameter of the truncated normal distribution.
           mean: The mean of the truncated normal distribution.
           lower: The lower bound for truncation.
           upper: The upper bound for truncation.
         """
        self.mean = mean
        self.stddev = stddev
        self.lower = lower
        self.upper = upper
        
    def __call__(self, shape: Sequence[int], dtype: Any) -> jnp.ndarray:
        m = jax.lax.convert_element_type(self.mean, new_dtype='float64')
        s = jax.lax.convert_element_type(self.stddev, new_dtype='float64')
        unscaled_r = jax.random.truncated_normal(hk.next_rng_key(), self.lower, self.upper, shape, dtype='float64')
        unscaled_c = jax.random.truncated_normal(hk.next_rng_key(), self.lower, self.upper, shape, dtype='float64')
        return jax.lax.complex(s * unscaled_r + m, s * unscaled_c + m)


class Cmplx_Xavier_Init(hk.initializers.Initializer):
    """Initialize complex magnitude according to the complex variant to the Glorot initialization
    proposed by Trabelsi (https://arxiv.org/abs/1705.09792). The magnitudes are sampled according to 
    Rayleigh distribution with scale parameter sigma=1/sqrt(n_in + n_out), while the phases are 
    uniform in [-pi, pi]."""

    def __init__(self, input_units: int, output_units: int):
        """ Constructs a :class:`Cmplx_Xavier_Init` initializer.

        Args:
           input_units: size of the input to the layer.
           output_units: number of neurons in the layer.
        """
        self.n_in = input_units
        self.n_out = output_units

    def __call__(self, shape: Sequence[int], dtype: Any) -> jnp.ndarray:

        # Jax does not support yet Rayleigh distribution
        sigma = 1 / np.sqrt(self.n_in + self.n_out)
        magnitudes = jnp.array( np.random.rayleigh(scale=sigma, size=shape) )

        phases = jax.random.uniform(hk.next_rng_key(), shape, dtype='float64', minval=-np.pi, maxval=np.pi)

        return magnitudes * jnp.exp( 1.j * phases )



class Cmplx_He_Init(hk.initializers.Initializer):
    """Initialize complex magnitude according to the complex variant to the He initialization
    proposed by Trabelsi (https://arxiv.org/abs/1705.09792). The magnitudes are sampled according to 
    Rayleigh distribution with scale parameter sigma=1/sqrt(n_in), while the phases are 
    uniform in [-pi, pi]."""

    def __init__(self, input_units: int):
        """ Constructs a :class:`Cmplx_He_Init` initializer.

        Args:
           input_units: size of the input to the layer.
        """
        self.n_in = input_units

    def __call__(self, shape: Sequence[int], dtype: Any) -> jnp.ndarray:

        # Jax does not support yet Rayleigh distribution
        sigma = 1 / np.sqrt(self.n_in)
        magnitudes = jnp.array( np.random.rayleigh(scale=sigma, size=shape) )

        phases = jax.random.uniform(hk.next_rng_key(), shape, dtype='float64', minval=-np.pi, maxval=np.pi)

        return magnitudes * jnp.exp( 1.j * phases )
