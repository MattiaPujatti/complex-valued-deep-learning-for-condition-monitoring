""" Definition of some complex activation functions according to the literature. """

import jax.numpy as jnp
import numpy as np
import torch
from jax import lax
from jax import custom_jvp

def cmplx_sigmoid(z):

    return 1. / ( 1. + jnp.exp(-z) )


def separable_sigmoid(z):

    return cmplx_sigmoid(jnp.real(z)) + 1.j*cmplx_sigmoid(jnp.imag(z))


def siglog(z):

    c = 1.   # stepness
    r = 1.   # scale

    return z / (c + 1./r*jnp.abs(z))


def igaussian(z):

    sigma = 1.
    g = 1 - jnp.exp( -z*jnp.conj(z) / (2*sigma**2) )
    n = z / jnp.sqrt( z*jnp.conj(z) )

    return g * n


@custom_jvp
def cardioid(z):
    return 0.5 * (1 + jnp.cos(jnp.angle(z))) * z

# df / dz
cardioid.defjvps(lambda g, ans, x: 0.5 + 0.5*jnp.cos(jnp.angle(x)) + 0.25j*jnp.sin(jnp.angle(x)))
