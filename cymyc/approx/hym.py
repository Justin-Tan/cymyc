r"""
Approximations of Hermitian Yang-Mills (HYM) metrics on line_bundles
"""
import jax
import jax.numpy as jnp
from jax import jit, jacfwd, vmap

import optax

from typing import List, Callable, Mapping, Tuple
from jaxtyping import Array, Float, DTypeLike

import numpy as np

from cymyc import curvature
from cymyc.utils import math_utils
from functools import partial


@partial(jax.jit, static_argnums=(1,2,3))
def reference_hermitian_structure(p: Float[Array, "i"], line_bundle: tuple, ambient: tuple, cdtype: DTypeLike = np.complex64):
    r"""Computes logarithm of the reference (Fubini--Study) Hermitian structure on the line bundle $O_X(k)$, 
    or direct sums thereof.
    """
    p = math_utils.to_complex(p).astype(cdtype)
    log_H = 0.
    for i in range(len(ambient)):
        s, e = np.sum(ambient[:i]).astype(np.int32) + i, np.sum(ambient[:i+1]) + i + 1
        z_i = jax.lax.dynamic_slice(p, (s,), (e-s,))
        kappa = jnp.real(jnp.sum(z_i * jnp.conj(z_i)))
        log_H += (-line_bundle[i]) * jnp.log(kappa)

    return log_H

@partial(jax.jit, static_argnums=(3,))
def connection_form(p, pullbacks, params, log_H_fn):
    # only for line bundles
    A = curvature.del_z(p, log_H_fn, params)
    return jnp.einsum("...a,...ia->...i", A, pullbacks)

@partial(jax.jit, static_argnums=(3,))
def curvature_form(p, pullbacks, params, log_H_fn):
    # only for line bundles
    ddbar_log_H = curvature.del_z_bar_del_z(p, log_H_fn, True, params)
    ddbar_log_H_pb = jnp.einsum("...ia,...jb,...ab->...ij", pullbacks, jnp.conj(pullbacks), ddbar_log_H)
    return ddbar_log_H_pb

@partial(jax.jit, static_argnums=(2,))
def connection_form_V(p, pullbacks, H_fn, params=None):
    H = H_fn(p)
    H_inv = jnp.linalg.inv(H)  # \bar{a} b
    if params is None:
        del_H = curvature.del_z(p, H_fn)
    else:
        del_H = curvature.del_z(p, H_fn, params)
    del_H = jnp.einsum("...abu,...iu->...abi", del_H, pullbacks)
    A = jnp.einsum("...bc, ...abi->...cai", H_inv, del_H)
    return A

@partial(jax.jit, static_argnums=(2,))
def curvature_form_V(p, pullbacks, H_fn, params=None):
    F = curvature.del_bar_z(p, connection_form_V, False, pullbacks, H_fn, params)
    F = jnp.einsum("...abiu, ...ju->...abij", F, jnp.conjugate(pullbacks))
    return F

@partial(jax.jit, static_argnums=(2,3,4))
def objective_function(data, params, curvature_form_fn, metric_fn, slope: float):
    p, pbs, w = data
    g = vmap(metric_fn)(p)  # frozen params
    F = vmap(curvature_form_fn, in_axes=(0, 0, None))(p, pbs, params)

    g_tr_F = -jnp.real(jnp.einsum("...ji,...ij->...", jnp.linalg.inv(g), F))  # why is this real?
    # return (w*(g_tr_F - slope)**2).sum() / w.sum()  # look at Ashmore paper
    return jnp.mean(w * (g_tr_F - slope)**2) / jnp.mean(w)  # look at Ashmore paper


@partial(jax.jit, static_argnums=(2,3,4))
def objective_function_implicit_slope(data, params, curvature_form_fn, metric_fn, d=1.):
    """
    Ref: (A7) https://arxiv.org/pdf/2110.12483 for d=1.
    """
    p, pbs, w = data
    g = vmap(metric_fn)(p)  # frozen params
    F = vmap(curvature_form_fn, in_axes=(0, 0, None))(p, pbs, params)

    g_tr_F = -jnp.real(jnp.einsum("...ji,...ij->...", jnp.linalg.inv(g), F))
    vol_Omega = jnp.mean(w)
    # return ((w*(g_tr_F**2)).sum() / w.sum()) - (w*g_tr_F).sum()**2 / w.sum()**2
    return jnp.mean(w * (g_tr_F**2)) / vol_Omega - 1./d * jnp.mean(w * g_tr_F)**2 / vol_Omega**2

def trace_F(data, params, curvature_form_fn, metric_fn):
    p, pbs, w = data
    g = vmap(metric_fn)(p)  # frozen params
    F = vmap(curvature_form_fn, in_axes=(0, 0, None))(p, pbs, params)

    g_tr_F = -jnp.real(jnp.einsum("...ji,...ij->...", jnp.linalg.inv(g), F))
    return g_tr_F

def loss_breakdown(data, params, curvature_form_fn, metric_fn, slope = None, d=1.):
    if slope is not None:
        loss = objective_function(data, params, curvature_form_fn, metric_fn, slope, d)
    else:
        loss = objective_function_implicit_slope(data, params, curvature_form_fn, metric_fn)

    p, pbs, w = data
    g_tr_F = trace_F(data, params, curvature_form_fn, metric_fn)
    return {'loss': loss, 'g_tr_F': jnp.mean(w * g_tr_F) / jnp.mean(w)}

@partial(jax.jit, static_argnums=(3,4,5,6))
def train_step(data, params, opt_state, optimizer, curvature_form_fn, metric_fn, slope: float = None):
    if slope is not None:
        loss, grads = jax.value_and_grad(objective_function, argnums=1)(data, params, curvature_form_fn, metric_fn, slope)
    else:
        loss, grads = jax.value_and_grad(objective_function_implicit_slope, argnums=1)(data, params, curvature_form_fn, metric_fn)
    param_updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, param_updates)
    return params, opt_state, loss
