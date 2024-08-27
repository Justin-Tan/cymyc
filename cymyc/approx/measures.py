import jax
import jax.numpy as jnp
from jax import vmap, jit
import numpy as np

from functools import partial

from .. import curvature

@partial(jit, static_argnums=(1,))
def sigma_measure(data, metric_fn):

    # weights: dVol_Omega / dVol_ref
    p, weights, dVol_Omega = data
    vol_Omega = jnp.mean(weights)
    
    g_herm = vmap(metric_fn)(p)
    det_g = jnp.real(jnp.linalg.det(g_herm))
    dVol_ratio = det_g / dVol_Omega
    vol_approx = jnp.mean(dVol_ratio * weights)
    vol_ratio = vol_Omega / vol_approx
        
    sigma_integrand = jnp.abs(1. - vol_ratio * dVol_ratio)
    sigma = 1./vol_Omega * jnp.mean(sigma_integrand * weights)
    return sigma

@partial(jit, static_argnums=(3,))
def ricci_measure(data, pullbacks, metric_fn, cy_dim):
    print(f'Compiling {ricci_measure.__qualname__}')
    p, weights, dVol_Omega = data

    vol_Omega = jnp.mean(weights)
    
    g_herm = vmap(metric_fn)(p)
    det_g = jnp.real(jnp.linalg.det(g_herm))
    dVol_ratio = det_g / dVol_Omega
    vol_approx = jnp.mean(dVol_ratio * weights)

    R = jnp.abs(vmap(curvature.ricci_scalar_from_form, in_axes=(0,None,0))(p, metric_fn, pullbacks))
    R_dVol_approx = R * dVol_ratio * weights
    _ricci_measure = (vol_approx ** (1./cy_dim) / vol_Omega) * jnp.mean(R_dVol_approx)
    return _ricci_measure
