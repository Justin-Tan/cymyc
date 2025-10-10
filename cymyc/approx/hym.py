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
def connection_form_line(p, pullbacks, params, log_H_fn):
    # only for line bundles
    A = curvature.del_z(p, log_H_fn, params)
    return jnp.einsum("...a,...ia->...i", A, pullbacks)

@partial(jax.jit, static_argnums=(3,))
def curvature_form_line(p, pullbacks, params, log_H_fn):
    # only for line bundles
    ddbar_log_H = curvature.del_z_bar_del_z(p, log_H_fn, True, params)
    ddbar_log_H_pb = jnp.einsum("...ia,...jb,...ab->...ij", pullbacks, jnp.conj(pullbacks), ddbar_log_H)
    return ddbar_log_H_pb

@partial(jax.jit, static_argnums=(2,))
def connection_form_V(p, pullbacks, H_fn, params=None):
    if params is None:
        H = H_fn(p)
        del_H = curvature.del_z(p, H_fn)
    else:
        H = H_fn(p, params)
        del_H = curvature.del_z(p, H_fn, False, params)
    H_inv = jnp.linalg.inv(H)  # \bar{a} b
    del_H = jnp.einsum("...abu,...iu->...abi", del_H, pullbacks)
    A = jnp.einsum("...bc, ...abi->...cai", H_inv, del_H)  # A^c_{ai}
    return A

@partial(jax.jit, static_argnums=(2,))
def curvature_form_V(p, pullbacks, H_fn, params=None):
    F = curvature.del_bar_z(p, connection_form_V, False, pullbacks, H_fn, params)
    F = jnp.einsum("...abiu, ...ju->...abij", F, jnp.conjugate(pullbacks))
    return F

@partial(jax.jit, static_argnums=(2,))
def _curvature_form_V(p, pullbacks, H_fn):
    F = curvature.del_bar_z(p, connection_form_V, False, pullbacks, H_fn)
    F = jnp.einsum("...abiu, ...ju->...abij", F, jnp.conjugate(pullbacks))
    return F

@partial(jax.jit, static_argnums=(2,3,4))
def objective_function(data, params, curvature_form_fn, metric_fn, slope: float):
    p, pbs, w = data
    g = vmap(metric_fn)(p)  # frozen params
    F = vmap(curvature_form_fn, in_axes=(0, 0, None))(p, pbs, params)

    # eigs of Hermitian matrix are real
    g_tr_F = -jnp.real(jnp.einsum("...ji,...ij->...", jnp.linalg.inv(g), F))
    # return (w*(g_tr_F - slope)**2).sum() / w.sum()  # look at Ashmore paper
    return jnp.mean(w * (g_tr_F - slope)**2) / jnp.mean(w)  # look at Ashmore paper


@partial(jax.jit, static_argnums=(2,3,4))
def objective_function_implicit_slope(data, params, curvature_form_fn, metric_fn, d=1.):
    """
    Ref: (A7) https://arxiv.org/pdf/2110.12483 for d=1.
    """
    p, pbs, w = data
    g = vmap(metric_fn)(p)  # frozen params
    # F = vmap(curvature_form_fn, in_axes=(0, 0, None))(p, pbs, params)
    F = vmap(curvature_form_fn, in_axes=(0, None))(p, params)

    # g_tr_F = -jnp.real(jnp.einsum("...ji,...ij->...", jnp.linalg.inv(g), F))
    g_tr_F = jnp.abs(jnp.einsum("...ji,...ij->...", jnp.linalg.inv(g), F))
    vol_Omega = jnp.mean(w)
    # return ((w*(g_tr_F**2)).sum() / w.sum()) - (w*g_tr_F).sum()**2 / w.sum()**2
    return jnp.mean(w * (g_tr_F**2)) / vol_Omega - 1./d * jnp.mean(w * g_tr_F)**2 / vol_Omega**2

@partial(jax.jit, static_argnums=(2,3,4))
def objective_function_implicit_slope_V(data, params, curvature_form_fn, metric_fn, 
        bundle_metric_fn, d=1.):
    """
    Ref: (A7) https://arxiv.org/pdf/2110.12483 for d=1.
    """
    p, pbs, w = data
    vol_Omega = jnp.mean(w)
    g = vmap(metric_fn)(p)  # frozen params
    g_inv = jnp.linalg.inv(g)
    # H = vmap(bundle_metric_fn, in_axes=(0,None))(p, params)
    F = vmap(curvature_form_fn, in_axes=(0, 0, None))(p, pbs, params)
    # F_up = jnp.einsum("...ji, ...kl, ...abik->...abjl", g_inv, g_inv, F) #  F^{\bar{\nu} \mu}^a_b
    # F_sq = jnp.einsum("...abij, ...cdij, ...db, ...ac->...", F, jnp.conjugate(F_up), jnp.linalg.inv(H), H)
    # F_sq = F_sq / jnp.mean(F_sq)
    #return jnp.mean(w * jnp.abs(F_sq)) / vol_Omega

    F_sq = jnp.einsum("...abij, ...bcij->...acij", F, F)
    tr_F_sq = jnp.einsum("...aaij->...ij", F_sq)
    g_tr_F_sq = jnp.einsum("...ji,...ij->...", g_inv, tr_F_sq)
    Tr_F_sq_g = g_tr_F_sq  # vmap(jnp.trace)(g_tr_F_sq)
    # return jnp.mean(w * jnp.abs(Tr_F_sq_g)) / vol_Omega

    # tr_F = jnp.einsum("...aaij->...ij", F)
    # g_tr_F = jnp.einsum("...ji,...ij->...", jnp.linalg.inv(g), tr_F)  # trace over base indices
    g_tr_F = jnp.einsum("...ji,...abij->...ab", jnp.linalg.inv(g), F)  # trace over base indices

    tr_F_g = vmap(jnp.trace)(g_tr_F)  # trace over fibre indices
    # tr_F_g_2 = vmap(jnp.trace)(g_tr_F @ jnp.einsum("...ab->...ba", jnp.conj(g_tr_F)))
    tr_F_g_2 = vmap(jnp.trace)(g_tr_F @ g_tr_F)
    #tr_F_g_2 = jnp.real(tr_F_g_2)
    tr_F_g_2 = jnp.real(tr_F_g_2)

    return jnp.mean(w * (tr_F_g_2)) / vol_Omega - 1./d * jnp.abs(jnp.mean(w * tr_F_g))**2 / vol_Omega**2

@partial(jax.jit, static_argnums=(2,3))
def trace_F(data, params, curvature_form_fn, metric_fn):
    p, pbs, w = data
    g = vmap(metric_fn)(p)  # frozen params
    F = vmap(curvature_form_fn, in_axes=(0, 0, None))(p, pbs, params)

    g_tr_F = -jnp.real(jnp.einsum("B...ji,B...ij->B...", jnp.linalg.inv(g), F))
    return g_tr_F

def loss_breakdown(data, params, curvature_form_fn, metric_fn, bundle_metric_fn = None, 
                   slope: float  = None, d: int = 1):
    if slope is not None:
        loss = objective_function(data, params, curvature_form_fn, metric_fn, slope, d)
    else:
        if d > 1:
            loss = objective_function_implicit_slope_V(data, params, curvature_form_fn, metric_fn, 
                    bundle_metric_fn, d)
        else:
            loss = objective_function_implicit_slope(data, params, curvature_form_fn, metric_fn)

    p, pbs, w = data
    if bundle_metric_fn is not None:
        H = vmap(bundle_metric_fn, in_axes=(0,None))(p, params)
    g_tr_F = trace_F(data, params, curvature_form_fn, metric_fn)
    det_g_tr_F = jnp.linalg.det(g_tr_F)
    max_eig = vmap(jnp.linalg.norm)(g_tr_F)
    vol_Omega = jnp.mean(w)
    if d > 1: g_tr_F = vmap(jnp.trace)(g_tr_F)

    return {'loss': loss, 'g_tr_F': jnp.mean(w * g_tr_F) / vol_Omega, "max_eig": jnp.mean(w * max_eig) / vol_Omega,
            'det_F_g': jnp.mean(w * det_g_tr_F) / vol_Omega, "det_H": jnp.mean(w * jnp.linalg.det(H)) / vol_Omega}

@partial(jax.jit, static_argnums=(3,4,5,6,7,8))
def train_step(data, params, opt_state, optimizer, curvature_form_fn, metric_fn, bundle_metric_fn, 
        rank_V: int = 1, slope: float = None):
    if slope is not None:
        loss, grads = jax.value_and_grad(objective_function, argnums=1)(data, params, curvature_form_fn, metric_fn, slope)
    else:
        if rank_V > 1:
            loss, grads = jax.value_and_grad(objective_function_implicit_slope_V, argnums=1)(data, params, 
                curvature_form_fn, metric_fn, bundle_metric_fn, rank_V)
        else:
            loss, grads = jax.value_and_grad(objective_function_implicit_slope, argnums=1)(data, params, 
                curvature_form_fn, metric_fn)
    param_updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, param_updates)
    return params, opt_state, loss

@partial(jax.jit, static_argnums=(3,4,))
def _train_step(data, params, opt_state, optimizer, objective_fn, aux_params=None):
    loss, grads = jax.value_and_grad(objective_fn, argnums=1)(data, params, aux_params)
    param_updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, param_updates)
    return params, opt_state, loss
