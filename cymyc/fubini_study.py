r"""Computation of the Fubini-Study metric - the unique $U(n+1)$ Riemannian metric on $\mathbb{P}^n$ + associated functions.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import jax
import jax.numpy as jnp
import numpy as np

from jax import jit
from functools import partial

from numpy.typing import DTypeLike
from typing import Callable, Sequence
from jaxtyping import Array, Float, Complex, ArrayLike

from . import alg_geo
from .utils import math_utils

@partial(jit, static_argnums=(2,))
def fubini_study_metric(p: Float[Array, "i"], normalization: Complex = jax.lax.complex(1.,0.), cdtype: DTypeLike = np.complex64):
    r"""Returns Fubini-Study metric in $\mathbb{P}^n$ evaluated at `p` in inhomogeneous coordinates.

    Parameters
    ----------
    p   :   array_like
        2*n real inhomogeneous coords at which metric is evaluated. Shape [i].
    Returns
    ----------
    g_FS    :   array_like
        Hermitian metric in local coordinates, $g_{\mu \bar{\nu}}$. Shape [i,j].
    """

    # Inhomogeneous coords
    complex_dim = p.shape[-1]//2
    zeta = jax.lax.complex(p[:complex_dim],
                           p[complex_dim:])
    zeta_bar = jnp.conjugate(zeta)
    zeta_sq = 1. + jnp.sum(zeta * zeta_bar)
    
    zeta_outer = jnp.einsum('...i,...j->...ij', zeta_bar, zeta)
    delta_mn = jnp.eye(complex_dim, dtype=cdtype) 

    g_FS = jnp.divide(delta_mn * zeta_sq - zeta_outer, jnp.square(zeta_sq))
    
    return g_FS * normalization / jnp.pi


@partial(jit, static_argnums=(2,))
def fubini_study_metric_homo(p: Float[Array, "i"], normalization: Complex = jax.lax.complex(1.,0.), cdtype: DTypeLike = np.complex64):
    r"""Returns Fubini-Study metric in $\mathbb{P}^n$ evaluated at `p` in homogeneous coordinates.

    Parameters
    ----------
    p   :   array_like
        2*(n+1) real homogeneous coords at which metric is evaluated. Shape [i].
    Returns
    ----------
    g_FS    :   array_like
        Hermitian metric in local coordinates, $g_{\mu \bar{\nu}}$. Shape [i,j].

        !!! warning
        Note the returned metric is expressed in homogeneous coordinates and will not be of full rank.
    """

    # Inhomogeneous coords
    complex_dim = p.shape[-1]//2
    zeta = jax.lax.complex(p[:complex_dim],
                           p[complex_dim:])
    zeta_bar = jnp.conjugate(zeta)
    zeta_sq = jnp.sum(zeta * zeta_bar)
    
    zeta_outer = jnp.einsum('...i,...j->...ij', zeta_bar, zeta)
    delta_mn = jnp.eye(complex_dim, dtype=cdtype) 

    g_FS = jnp.divide(delta_mn * zeta_sq - zeta_outer, jnp.square(zeta_sq))
    
    return g_FS * normalization / jnp.pi

@partial(jit, static_argnums=(2,3,4,5))
def fubini_study_metric_homo_pb(p: Float[Array, "i"], dQdz_info: tuple, cy_dim: int, 
                                normalization: Complex = jax.lax.complex(1.,0.),
                                ambient_out: bool = False, cdtype: DTypeLike = np.complex64):
    r"""Returns FS metric on hypersurfaces $X$ immersed in $\mathbb{P}^n$ evaluated 
    at `p` in homogeneous coordinates, i.e. $[x_1 : x_2: \cdots : x_{n+1}]$. This is the 
    ambient FS metric in CP^n pulled back by the inclusion map: $\iota: X \hookrightarrow \mathbb{P}^n$. 
    Parameters
    ----------
    p : array_like
        2*(n+1) real homogeneous coords at which metric is evaluated. Shape [i].
    Returns
    ----------
    g_FS_pb : array_like
        Hermitian metric pulled back to $X$ in local coordinates, $g_{\mu \bar{\nu}}$. Shape [i,j].
    """
    g_FS = fubini_study_metric_homo(p, normalization, cdtype=cdtype)
    pullbacks = alg_geo.compute_pullbacks(math_utils.to_complex(p), dQdz_info, cy_dim, cdtype=cdtype)

    if ambient_out is True: return g_FS, pullbacks

    g_FS_pb = jnp.einsum('...ia,...ab,...jb->...ij', pullbacks, g_FS, 
        jnp.conjugate(pullbacks))
    
    return g_FS_pb

@partial(jit, static_argnums=(2,3,5))
def fubini_study_metric_homo_pb_cicy(p: Float[Array, "i"], pullbacks: Complex[Array, "cy_dim i"], n_coords : int,
                                     ambient : tuple, k_moduli: Array = None, cdtype: DTypeLike = np.complex64):
    r"""
    Returns ambient Fubini-Study metric for a CICY in product of projective spaces pulled back by the inclusion map
    $\iota: X \hookrightarrow \mathbb{P}^{n_1} \times \cdots \times \mathbb{P}^{n_K}$.

    Parameters
    ----------
    p   :   array_like
        2*(n+1) real homogeneous coords at which metric is evaluated. Shape [i].
    pullbacks : array_like
        Pullback tensor from ambient to projective variety.
    n_coords : int
        Dimension of ambient combined projective space.
    ambient : tuple
        Dimension of each projective space factor.
    k_moduli : array_like, optinal
        Kahler moduli for each projective space factor.
    Returns
    ----------
    g_FS_pb    :  array_like
        Hermitian metric pulled back to $X$ in local coordinates, $g_{\mu \bar{\nu}}$. Shape [i,j].

        !!! warning
        Note the returned metric is expressed in homogeneous coordinates and will not be of full rank.
    """

    p = math_utils.to_complex(p)
    g_FS = jnp.zeros((n_coords, n_coords), dtype=cdtype)

    if k_moduli is None: k_moduli = jnp.ones_like(np.array(ambient), dtype=cdtype)

    for i in range(len(ambient)):
        s, e = np.sum(ambient[:i]).astype(np.int32) + i, np.sum(ambient[:i+1]) + i + 1
        p_ambient_i_homo = math_utils.to_real(jax.lax.dynamic_slice(p, (s,), (e-s,)))
        g_FS_ambient_i_homo = fubini_study_metric_homo(p_ambient_i_homo, k_moduli[i], cdtype=cdtype)
        g_FS = jax.lax.dynamic_update_slice(g_FS, g_FS_ambient_i_homo, (s,s))

    g_FS_pb = jnp.einsum('...ia,...ab,...jb->...ij', pullbacks, g_FS,
        jnp.conjugate(pullbacks))

    return g_FS_pb

@partial(jit, static_argnums=(2,3,5))
def det_fubini_study_pb_cicy(p, pullbacks, n_coords, ambient, k_moduli=None, cdtype=np.complex64):
    # r"""Wrapper function for determinant of output of fubini_study_metric_homo_pb_cicy.
    # """
    g_FS_pb = fubini_study_metric_homo_pb_cicy(p, pullbacks, n_coords, ambient, k_moduli, cdtype)
    det_pb = jnp.linalg.det(jnp.squeeze(g_FS_pb))
    return jnp.real(det_pb)

@partial(jit, static_argnums=(3,4,5,6,8,9))
def _fubini_study_metric_homo_gen_pb_cicy(p: Float[Array, "i"], dQdz_monomials: Float[Array, "_"], 
                                          dQdz_coeffs: Complex[Array, "_"], n_hyper: int, cy_dim: int, n_coords: int,
                                          ambient: tuple, k_moduli: Array = None, ambient_out: bool = False,
                                          cdtype: DTypeLike = np.complex64):
    r"""
    Generates pullbacks and returns ambient Fubini-Study metric for general CICY
    in product of projective spaces pulled back by the inclusion map $\iota: X \righthookarrow \mathbb{P}^{n_1} \times \cdots \times \mathbb{P}^{n_K}$.

    Parameters
    ----------
    p   :   array_like
        2*(n+1) real homogeneous coords at which metric is evaluated. Shape [i].
    pullbacks : array_like
        Pullback tensor from ambient to projective variety. If supplied, computes Ricci tensor
        on the variety.
    n_coords : int
        Dimension of ambient combined projective space.
    ambient : tuple
        Dimension of each projective space factor.
    k_moduli : array_like, optinal
        Kahler moduli for each projective space factor.
    Returns
    ----------
    g_FS_pb    :  array_like
    Hermitian metric pulled back to $X$ in local coordinates, $g_{\mu \bar{\nu}}$. Shape [i,j].

        !!! warning
        Note the returned metric is expressed in homogeneous coordinates and will not be of full rank.
    """

    p = math_utils.to_complex(p)
    g_FS = jnp.zeros((n_coords, n_coords), dtype=cdtype)

    if k_moduli is None:
        k_moduli = jnp.ones_like(np.array(ambient), dtype=cdtype)
    else:
        k_moduli = k_moduli.astype(cdtype)

    for i in range(len(ambient)):
        s, e = np.sum(ambient[:i]).astype(np.int32) + i, np.sum(ambient[:i+1]) + i + 1
        p_ambient_i_homo = math_utils.to_real(jax.lax.dynamic_slice(p, (s,), (e-s,)))
        g_FS_ambient_i_homo = fubini_study_metric_homo(p_ambient_i_homo, k_moduli[i], cdtype=cdtype)
        g_FS = jax.lax.dynamic_update_slice(g_FS, g_FS_ambient_i_homo, (s,s))

    pullbacks = alg_geo._pullbacks_cicy(p, dQdz_monomials, dQdz_coeffs, n_hyper, cy_dim, n_coords, aux=False,
            cdtype=cdtype)

    if ambient_out is True: return g_FS, pullbacks

    g_FS_pb = jnp.einsum('...ia,...ab,...jb->...ij', pullbacks, g_FS,
        jnp.conjugate(pullbacks))

    return g_FS_pb


@partial(jit, static_argnums=(2,3))
def fubini_study_metric_homo_pb_precompute(p, pullbacks, normalization=jax.lax.complex(1.,0.), 
                                           cdtype: DTypeLike = np.complex64):
    # r"""Identical to `fubini_study_metric_homo_pb` but expects pullbacks as additional argument.
    # """
    g_FS = fubini_study_metric_homo(p, normalization, cdtype)
    g_FS_pb = jnp.einsum('...ia,...ab,...jb->...ij', pullbacks, g_FS, 
        jnp.conjugate(pullbacks))
    return g_FS_pb

@jit
def _det_fubini_study_CP_n(p: Float[Array, "i"]):
    r"""Returns determinant of Fubini-Study metric in $\mathbb{P}^n$ evaluated at `p` 
    in inhomogeneous coordinates.

    Parameters
    ----------
    p : array_like
        2*n real inhomogeneous coords at which metric is evaluated. Shape [i].
    Returns
    ----------
    det_g_FS    :   array_like
        Determinant of Hermitian metric in local coordinates. Shape [...]
    """  
    complex_dim = p.shape[-1]//2
    zeta_sq = 1. + jnp.sum(p**2)
    
    return 1./(zeta_sq)**(complex_dim + 1)

@partial(jit, static_argnums=(2,))
def det_fubini_study_pb(p: Float[Array, "i"], pullbacks: Complex[Array, "cy_dim i"], 
                        cdtype: DTypeLike = np.complex64):
    # r"""Wrapper function for determinant of Fubini-Study metric pulled back to hypersurfaces.
    # """
    g_FS = fubini_study_metric_homo(p, cdtype=cdtype)
    g_FS_pb = jnp.einsum('...ia,...ab,...jb->...ij', pullbacks, g_FS, 
        jnp.conjugate(pullbacks))
    det_pb = jnp.linalg.det(g_FS_pb)
    return jnp.real(det_pb)

@partial(jit, static_argnums=(1,))
def fubini_study_inverse(p: Float[Array, "i"], cdtype: DTypeLike = np.complex64):
    r"""Returns analytic inverse in inhomogeneous coords using the Woodbury matrix identity.

    Parameters
    ----------
    p : array_like
        2*n real inhomogeneous coords at which metric is evaluated. Shape [i].
    Returns
    ----------
    g_FS_inv    :   array_like
        Inverse of Hermitian metric in inhomogeneous coordinates. Shape [i,j]
    """

    complex_dim = p.shape[-1]//2
    zeta = jax.lax.complex(p[:complex_dim],
                           p[complex_dim:])
    zeta_bar = jnp.conjugate(zeta)
    zeta_outer = jnp.einsum('...i,...j->...ij', zeta_bar, zeta)
    delta_mn = jnp.eye(complex_dim, dtype=cdtype)
    return (1. + jnp.sum(p**2)) * (zeta_outer + delta_mn)

@jit
def CP_n_fs_ricci_form(p):
    complex_dim = p.shape[-1]//2
    return (complex_dim + 1) * 1.j * fubini_study_metric(p)


def _fs_metric(p, normalization=jax.lax.complex(1.,0.), cdtype=np.complex64):
    r"""
    Internal use. Returns FS metric in a single projective space CP^n evaluated at `p` 
    in inhomogeneous coordinates.
    Parameters
    ----------
        `p`     : Complex `n`-dim inhomogeneous coords at 
                    which metric matrix is evaluated. Shape [i].
    Returns
    ----------
        `g`     : Hermitian metric in CP^n, $g_{\mu \bar{\nu}}$. Shape [i,j].
    """
    complex_dim, zeta = p.shape[-1], p
    zeta_bar = jnp.conjugate(zeta)
    zeta_sq = 1. + jnp.sum(zeta * zeta_bar)
    zeta_outer = jnp.einsum('...i,...j->...ij', zeta_bar, zeta)
    delta_mn = jnp.eye(complex_dim, dtype=cdtype) 

    g_FS = jnp.divide(delta_mn * zeta_sq - zeta_outer, jnp.square(zeta_sq))
    
    return g_FS * normalization

def _fs_metric_inverse(p, normalization=jax.lax.complex(1.,0.), cdtype=np.complex64):
    r"""
    Returns FS metric inverse in single projective space $\mathbb{P}^n$ evaluated at `p` 
    in inhomogeneous coordinates using the Woodbury matrix identity.
    Notes
    -----
    Note this returns $g^{\bar{\nu} \mu}$, the inverse operation is
    $g^{\bar{\nu} \mu}g_{\mu \bar{\kappa}} = \delta^{\bar{\nu}}_{\bar{\kappa}}$. 
    Let `g_inv` be the output of this fn, then `g_inv @ g = jnp.eye(n)` and 
    $g^{\mu \bar{\nu}}$ = `jnp.conjugate(g_inv)`.
    """
    complex_dim, zeta = p.shape[-1], p
    zeta_bar = jnp.conjugate(zeta)
    zeta_sq = 1. + jnp.sum(zeta * zeta_bar)
    zeta_outer = jnp.einsum('...i,...j->...ij', zeta_bar, zeta)
    delta_mn = jnp.eye(complex_dim, dtype=cdtype) 
    return zeta_sq * (zeta_outer + delta_mn) / normalization

def _fs_metric_homo_cicy(p, n_inhomo_coords, ambient, normalization=jax.lax.complex(1.,0.), 
        cdtype=np.complex64):
    r"""
    Returns ambient FS metric evaluated in product of projective spaces,
    P^{k_1}_1 \times P^{k_2}_2 \times \cdots \times P^{k_n}_n,
    returned in inhomogeneous coordinates.
    Parameters
    ----------
        `p`     : Complex `n+1`-dim homogeneous coords at 
                    which metric matrix is evaluated. Shape [i].
    """
    g_FS = jnp.zeros((n_inhomo_coords, n_inhomo_coords), dtype=cdtype)
    for i in range(len(ambient)):
        pt_s, pt_e = np.sum(ambient[:i]) + i, np.sum(ambient[:i+1]) + i + 1
        g_s, g_e = np.sum(ambient[:i]), np.sum(ambient[:i+1])

        p_ambient_i = jax.lax.dynamic_slice(p, (pt_s,), (pt_e-pt_s,))
        p_ambient_i_inhomo = math_utils._inhomogenize(p_ambient_i)
        g_FS_ambient_i = _fs_metric(p_ambient_i_inhomo, normalization, cdtype)
        g_FS = jax.lax.dynamic_update_slice(g_FS, g_FS_ambient_i, (g_s, g_s))

    return g_FS
