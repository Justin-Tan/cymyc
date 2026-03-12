r"""
Approximations of Harmonic line-bundle valued 1-forms.
"""
import jax
import jax.numpy as jnp
from jax import vmap, jit

import optax

from cymyc.curvature import del_bar_z, del_z_bar_del_z
from functools import partial

@partial(jax.jit, static_argnums=(2,))
def objective_fn(data, params, s_fn):
    s_fn = partial(s_fn, params = params)

    # unroll
    p, pbs, w, g, A, d_dual_ref = data

    ds_dzbar = jax.vmap(partial(del_bar_z, fun=s_fn))(p)
    ds_dzbar = jnp.einsum("...ai,...i->...a", jnp.conj(pbs), ds_dzbar)

    dds_real = jax.vmap(partial(del_z_bar_del_z, fun = lambda x: jnp.real(s_fn(x))))(p)
    dds_imag = jax.vmap(partial(del_z_bar_del_z, fun = lambda x: jnp.imag(s_fn(x))))(p)
    dds = jnp.einsum("...ai,...bj,...ij->...ab", pbs, jnp.conj(pbs), dds_real + 1j*dds_imag)

    d_dual_ds = -jnp.einsum("...ji,...ij->...", jnp.linalg.inv(g), dds + A[..., jnp.newaxis]*ds_dzbar[..., jnp.newaxis, :])

    return jnp.real(jnp.mean(w*jnp.power(jnp.abs(d_dual_ds + d_dual_ref), 2)) / jnp.mean(w))


@partial(jax.jit, static_argnums=(3,4))
def train_step(data, params, opt_state, s_fn, optimizer):
    loss, grads = jax.value_and_grad(objective_fn, argnums=1)(data, params, s_fn)
    param_updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, param_updates)
    return params, opt_state, loss