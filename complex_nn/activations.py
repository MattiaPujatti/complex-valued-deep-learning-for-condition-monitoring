""" Definition of some complex activation functions according to the literature. """

import jax
import jax.numpy as jnp
from jax import lax, custom_jvp

def cmplx_sigmoid(z):
    """Implementation of the complex sigmoid."""
    return 1. / ( 1. + jnp.exp(-z) )

def cmplx_tanh(z):
    """Implementation of the complex hyperbolic tangent."""
    return jnp.tanh(z)

def separable_sigmoid(z):
    """Implementation of the separable sigmoid.
    Source: [Nitta et al. `An Extension of the Back-Propagation Algorithm to Complex Numbers`, (1997)]"""
    return cmplx_sigmoid(jnp.real(z)) + 1.j*cmplx_sigmoid(jnp.imag(z))

def separable_tanh(z):
    """Implementation of the separable sigmoid.
    Source: [Nitta et al. `An Extension of the Back-Propagation Algorithm to Complex Numbers`, (1997)]"""
    return cmplx_tanh(jnp.real(z)) + 1.j*cmplx_tanh(jnp.imag(z))

def CReLU(z):
    """Implementation of the CReLU activation function, i.e. separable ReLU.
    Source: [Trabelsi et al. `Deep complex networks`, (2018)]"""
    return jax.nn.relu(jnp.real(z)) + 1.j*jax.nn.relu(jnp.imag(z))

def zReLU(z):
    """Implementation of the zReLU activation function, i.e. a more restrictive separable ReLU.
    Source: [Guberman et al., `On complex valued convolutional neural networks`, (2016)]"""
    on_condition = (jnp.angle(z) >= 0 and jnp.angle(z) <= jnp.pi/2.)
    return jnp.where(on_condition, z, 0.+0.j)

def siglog(z, c=1., r=1.):
    """Implementation of the siglog activation function:
       siglog(z) = sigmoid(log|z|)(z/|z|)  --> siglog(z) = z / ( c + (1/r)*|z| )
    c is the stepness and r the scale.
    Source: [Georgiou and Koutsougeras, `Complex domain backpropagation`, (1992)]"""
    return z / (c + (1./r)*jnp.abs(z))

def igaussian(z, sigma=1.):
    """Implementation of the iGaussian activation function:
       iGauss(z) = ( 1 - exp(-|z|**2 / (2sigma**2))(z/|z|)
    sigma is the parameter (std) of the reversed gaussian.
    Source: [Virtue, `Complex-valued Deep Learning with Applications to Magnetic Resonance Image Synthesis`, (2019)]"""
    g = 1 - jnp.exp( -z*jnp.conj(z) / (2*sigma**2) )
    n = z / jnp.absolute(z)
    return g * n

def modReLU(z, b=1.):
    """Implementation of the modReLU activation function:
       modReLU(z) = ReLU(|z| + b)e**(itheta_z)
    the parameter b represent the distance from the origin inside which the neuron is off.
    Source: [Arjovsky et al., `Unitary evolution recurrent neural networks`, (2015)]"""
    norm = z / jnp.absolute(z)
    return jax.nn.relu( jnp.absolute(z) + b ) * norm


@custom_jvp
def cardioid(z):
    """Implementation of the complex cardioid activation function:
       cardioid(z) = (1/2) * (1+cos(theta_z)) * z
    Source: [Virtue, `Complex-valued Deep Learning with Applications to Magnetic Resonance Image Synthesis`, (2019)]"""
    return 0.5 * (1 + jnp.cos(jnp.angle(z))) * z

# df / dz
cardioid.defjvps(lambda g, ans, x: 0.5 + 0.5*jnp.cos(jnp.angle(x)) + 0.25j*jnp.sin(jnp.angle(x)))
