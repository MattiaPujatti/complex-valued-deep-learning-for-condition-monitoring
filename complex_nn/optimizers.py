import jax.numpy as jnp
from jax.experimental.optimizers import optimizer, make_schedule


@optimizer
def cmplx_adam(step_size, b1=0.9, b2=0.999, eps=1e-8):
    step_size = make_schedule(step_size)
    def init(x0):
        m0 = jnp.zeros_like(x0)
        v0 = jnp.zeros_like(x0)
        return x0, m0, v0
    def update(i, g, state):
        x, m, v = state
        m = (1 - b1) * g + b1 * m  # First moment estimate.
        v = (1 - b2) * jnp.array(g)*jnp.conjugate(g) + b2 * v  # Second moment estimate.
        mhat = m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))  # Bias correction.
        vhat = v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
        x = x - step_size(i) * mhat / (jnp.sqrt(vhat) + eps)
        return x, m, v
    def get_params(state):
        x, _, _ = state
        return x
    return init, update, get_params
