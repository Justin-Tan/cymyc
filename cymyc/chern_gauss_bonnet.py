"""Calculation of topological invariants (Chern classes, Euler characteristic) for a general Kahler manifold.
Note some functions may specialise to the case of a projective variety. Note the following:

* As with curvature, these functions involve $n$-th order derivatives of some function `fun`. These are 
computed with autodiff - a good usage pattern is to make a partial closure to bind all arguments to `fun`
except the coordinate dependence.
* These functions expect local coordinates `z` in a `c_dim`-dimensional space, with the 
real and imaginary parts concatenated to form a real-valued `2*c_dim` vector, `p = [Re(z); Im(z)]`.
* The Euler characteristic is defined up to an integer factor from the normalisation of the volume form. The
canonical choice is to normalise according to the intersection number computation.
"""

import jax
import numpy as np
import math
import jax.numpy as jnp

from jax import jit

from functools import partial
from typing import Callable, Sequence, Optional, Tuple
from jaxtyping import Array, Float, Complex, ArrayLike

# custom
from . import curvature
from .utils import math_utils

eps_2d = jnp.array(math_utils.n_dim_eps_symbol(2))
eps_3d = jnp.array(math_utils.n_dim_eps_symbol(3))
eps_4d = jnp.array(math_utils.n_dim_eps_symbol(4))
eps_6d = jnp.array(math_utils.n_dim_eps_symbol(6))

def prefactor(n): return 1./((4*np.pi)**n * math.factorial(n))

@partial(jit, static_argnums=(1,))
def riem_real(p: Float[Array, "2*i"], metric_fn: Callable[[Array], Array], *args: Sequence, 
              pullbacks: Optional[Float[Array, "dim i"]] = None, return_down: bool = False) -> Float[Array, "2*dim 2*dim 2*dim 2*dim"]:
    """
    Viewing the $n$-dim Kähler manifold as a real $(2n)$-dimensional manifold, computes the real 
    $(2n)$-dimensional Riemann tensor corresponding to the given metric tensor.

    Parameters
    ----------
    p : array_like
        2 * `complex_dim` real coords with `float` type at which `fun` is evaluated. 
        Consists of the concatenation of real and imaginary parts along the last axis.
    metric_fn : callable
        Function representing metric tensor in local coordinates.
    *args : tuple
        Additional arguments to pass to `metric_fn`.
    pullbacks : array_like, optional
        Pullback matrix from ambient to projective variety. If supplied, computes Riemann tensor on the
        variety, by default None.
    return_down : bool, optional
        If True, return the Riemann tensor with all indices lowered, by default False.

    Returns
    -------
    array_like
        The real (2n)-dimensional Riemann tensor. If `return_down` is True, returns the tensor with all indices lowered.
    """ 
    g = metric_fn(p)  # n-dim hermitian part of metric
    dim = g.shape[-1]
    
    # Real (2n)-dim metric
    g_ab = jnp.zeros((2*dim, 2*dim), dtype=jnp.complex64)
    g_ab = g_ab.at[:dim, dim:].set(g)
    g_ab = g_ab.at[dim:, :dim].set(g.T)
    g_ab_inv = jnp.linalg.inv(g_ab)
    
    # $ R^{\kappa}_{\lambda \mu \bar{\nu}},  R^{\kappa}_{\lambda \bar{\mu} \nu} $, respectively.
    riem_ikjl, riem = curvature.riemann_tensor_kahler(p, metric_fn, pullbacks, return_aux=True)
    
    # R^a_{bcd}  raised `a` INDEX
    riem_abcd = jnp.zeros((2*dim, 2*dim, 2*dim, 2*dim), dtype=jnp.complex64)
    
    riem_abcd = riem_abcd.at[:dim, :dim, dim:, :dim].set(riem)
    riem_abcd = riem_abcd.at[dim:, dim:, :dim, dim:].set(jnp.conjugate(riem))
    riem_abcd = riem_abcd.at[:dim, :dim, :dim, dim:].set(riem_ikjl)
    riem_abcd = riem_abcd.at[dim:, dim:, dim:, :dim].set(jnp.conjugate(riem_ikjl))
    
    riem_abcd_down = jnp.einsum('...ai, ...abcd -> ...ibcd', g_ab, riem_abcd)
    riem_ab_up_cd_down = jnp.einsum('ai, bj, ijkl->abkl', g_ab_inv, g_ab_inv, riem_abcd_down)
    
    if return_down == True:
        return riem_abcd_down

    return riem_ab_up_cd_down

@partial(jit, static_argnums=(1,))
def euler_density_2d(p, metric_fn, pullbacks=None):
    # This has issues for the CP^1 case - TODO stop squeezing del_z, del_z_bar
    riem_ab_up_cd_down = riem_real(p, metric_fn, pullbacks)
    integrand = jnp.einsum('abcd, ab, cd->',riem_ab_up_cd_down, eps_2d, eps_2d)
    dim = riem_ab_up_cd_down.shape[-1]
    return integrand * prefactor(dim//2)

@partial(jit, static_argnums=(1,))
def euler_density_4d(p, metric_fn, pullbacks=None):
    riem_ab_up_cd_down = riem_real(p, metric_fn, pullbacks)
    integrand = jnp.einsum('abwx, cdyz, abcd, wxyz->',riem_ab_up_cd_down, riem_ab_up_cd_down, 
                eps_4d, eps_4d)
    dim = riem_ab_up_cd_down.shape[-1]
    return integrand * prefactor(dim//2)

@partial(jit, static_argnums=(1,))
def euler_density_6d(p, metric_fn, pullbacks=None): # fubini_study.fubini_study_metric
    riem_ab_up_cd_down = riem_real(p, metric_fn, pullbacks=pullbacks)
    integrand = jnp.einsum('abuv, cdwx, efyz, abcdef, uvwxyz->',
            riem_ab_up_cd_down, riem_ab_up_cd_down, riem_ab_up_cd_down,
            eps_6d, eps_6d)
    dim = riem_ab_up_cd_down.shape[-1]
    return integrand * prefactor(dim//2)

@partial(jit, static_argnums=(1,3))
def euler_density(p, metric_fn, pullbacks=None, cy_dim=3): # fubini_study.fubini_study_metric

    if cy_dim == 2:
        return euler_density_4d(p, metric_fn, pullbacks)

    return euler_density_6d(p, metric_fn, pullbacks)

@jit
def chern1(riem: Complex[Array, "dim dim dim dim"]) -> Complex[Array, "dim dim"]:
    r"""Computes the first Chern class. Let $\mathcal{R} \in \Omega^2_X\left(\text{End}(T_X)\right)$ be the curvature two-form on $X$, then
    $$ c_1 \propto \textsf{Tr}\mathcal{R}~.$$

    Parameters
    ----------
    riem : array_like
        (1,3) Riemann tensor corresponding to the Kahler connection 
        $R^{\kappa}_{\lambda \mu \bar{\nu}}$.

    Returns
    -------
    array_like
        First Chern form.
    """
    return 1.j/(2*np.pi) * jnp.einsum('...aaij->...ij', riem)

@jit
def chern2(riem: Complex[Array, "dim dim dim dim"]) -> Complex[Array, "dim dim dim dim"]:
    r"""Computes the second Chern class,
    $$ c_2 \propto \left(\textsf{Tr}\mathcal{R}^2 - \textsf{Tr}\mathcal{R} \wedge \textsf{Tr}\mathcal{R} \right)~.$$

    Parameters
    ----------
    riem : array_like
        (1,3) Riemann tensor corresponding to the Kahler connection 
        $R^{\kappa}_{\lambda \mu \bar{\nu}}$.

    Returns
    -------
    c2 : array_like
        Second Chern form.
    """
    TrR_w_TrR = jnp.einsum('...aaij, ...bbkl->...ijkl', riem, riem)
    TrR_sq = jnp.einsum('...abij, ...bakl->...ijkl', riem, riem)

    c2 = 1./(2*np.pi)**2 * 0.5 * (TrR_sq - TrR_w_TrR)
    return c2

@jit
def chern3(riem: Complex[Array, "dim dim dim dim"]) -> Complex[Array, ""]:
    r"""Computes the third Chern class,

    $$ c_3 \propto c_1 \wedge c_2 + c_1 \wedge \textsf{Tr} \mathcal{R}^2 - \textsf{Tr}\mathcal{R}^3~. $$

    Parameters
    ----------
    riem : array_like
        (1,3) Riemann tensor corresponding to the Kahler connection 
        $R^{\kappa}_{\lambda \mu \bar{\nu}}$.

    Returns
    -------
    c3 : array_like
        The third Chern form, expressed as the coefficient of the complex wedgey part 
        in standard form: $dz^1 \wedge d\bar{z}^1 \wedge \cdots \wedge dz^n \wedge d\bar{z}^n$.
    """
    
    c1, c2 = chern1(riem), chern2(riem)
    TrR_sq = jnp.einsum('...abij, ...bakl->...ijkl', riem, riem)
    TrR_cu = jnp.einsum('...abix,...bcjy,...cakz->...ixjykz', riem, riem, riem)

    c1_w_c2 = jnp.einsum('...ab,...cdef->...abcdef', c1, c2)
    c1_w_TrR_sq = jnp.einsum('...ab, ...cdef->...abcdef', c1, TrR_sq)
    
    _c3 = 1./3 * c1_w_c2 + 1./(2*np.pi**3) * 1./3 * c1_w_TrR_sq - (1./3) * (1.j / (2*np.pi)**3) * TrR_cu
    c3 = jnp.einsum('...ijk,...xyz,...ixjykz->...', eps_3d, eps_3d, _c3)
    return c3

@jit
def chern4(riem: Complex[Array, "dim dim dim dim"]) -> Complex[Array, ""]:
    r"""Computes the fourth Chern class,

    $$ c_4 \propto c_1^4 + c_1^2 \wedge \textsf{Tr}{\mathcal{R}}^2 + c_1 \wedge \textsf{Tr} \mathcal{R}^3 \
          +  (\textsf{Tr}\mathcal{R}^2)^2 - \textsf{Tr}\mathcal{R}^4~. $$

    Parameters
    ----------
    riem : array_like
        (1,3) Riemann tensor corresponding to the Kahler connection 
        $R^{\kappa}_{\lambda \mu \bar{\nu}}$.

    Returns
    -------
    c4 : array_like
        The fourth Chern form, expressed as the coefficient of the complex wedgey part 
        in standard form: $dz^1 \wedge d\bar{z}^1 \wedge \cdots \wedge dz^n \wedge d\bar{z}^n$.
    """
    
    c1 = chern1(riem)
    c1_sq = jnp.einsum('...ab, ...cd->...abcd', c1, c1)
    c1_qu = jnp.einsum('...ab, ...cd, ...ef, ...gh->...abcdefgh', c1, c1, c1, c1)

    TrR_sq = jnp.einsum('...abij, ...bakl->...ijkl', riem, riem)
    TrR_sq_w_TrR_sq = jnp.einsum('...ijkl, ...abcd->...ijklabcd', TrR_sq, TrR_sq)
    TrR_cu = jnp.einsum('...abix, ...bcjy,...cakz->...ixjykz', riem, riem, riem)
    TrR_qu = jnp.einsum('...abiw, ...bcjx, ...cdky, ...dalz->...iwjxkylz', riem, riem, riem, riem)

    c1_sq_w_TrR_sq = jnp.einsum('...abcd, ...ijkl->...abcdijkl', c1_sq, TrR_sq)
    c1_w_TrR_cu = jnp.einsum('...ab, ...cdefgh->...abcdefgh', c1, TrR_cu)

    _tmp1 = (c1_qu - 6./(2*np.pi)**2 * c1_sq_w_TrR_sq - 8.j/(2*np.pi)**3 *  c1_w_TrR_cu)
    _tmp2 = (TrR_sq_w_TrR_sq - 2. * TrR_qu)

    _c4 = 1./24 * _tmp1 + 1./(2*np.pi**4) * 1./8 * _tmp2    
    c4 = jnp.einsum('...ijkl,...wxyz,...iwjxkylz->...', eps_4d, eps_4d, _c4)
    return c4


@partial(jit, static_argnums=(3,))
def euler_characteristic(data: Tuple[Float[Array, "2 * i"], Float[Array, ""], Float[Array, ""]], 
                         pullbacks: Optional[Float[Array, "dim i"]], 
                         metric_fn: Callable[[Array], Array], cy_dim: int = 3) -> Float[Array, ""]:
    r"""
    Computes the Euler characteristic from the Pfaffian of the curvature two-form $\mathcal{R} \in \Omega^2_X(T_X)$,
    with support for Calabi-Yau $n$-folds for $n=1,2,3$.

    $$\chi = \frac{1}{(2\pi)^n} \int_X \textsf{Pf}(\mathcal{R})~.$$

    Parameters
    ----------
    data : tuple
        A tuple containing coordinates `p`, weights, and the volume form `dVol_Omega`.

    pullbacks : array_like, optional
        Pullback matrix from ambient to projective variety. If supplied, computes the Euler characteristic 
        on the variety, by default None.
    metric_fn : callable
        Function representing metric tensor in local coordinates.
    cy_dim : int, optional
        Complex dimension of Calabi-Yau, default 3.

    Returns
    -------
    chi: array-like
        The Euler characteristic.
    """
    p, weights, dVol_Omega = data
    p = math_utils.to_real(p)
    ec_integrand = euler_density(p, metric_fn, pullbacks, cy_dim)
    norm_factor = math.factorial(cy_dim)
    
    det_g = jnp.real(jnp.linalg.det(metric_fn(p)))
    chi = norm_factor * weights / dVol_Omega * det_g * ec_integrand 

    return chi

@partial(jit, static_argnums=(3,))
def euler_characteristic_form(data: Tuple[Float[Array, "2 * i"], Float[Array, ""], Float[Array, ""]], 
                              pullbacks: Optional[Float[Array, "dim i"]], 
                              metric_fn: Callable[[Array], Array], cy_dim: int = 3) -> Float[Array, ""]:
    r"""
    Computes the Euler characteristic via integration of the top Chern class over $X$ for a Calabi-Yau
    n-fold.

    $$\chi = \int_X c_n~.$$

    Parameters
    ----------
    data : tuple
        A tuple containing coordinates `p`, weights, and the volume form `dVol_Omega`.
    pullbacks : array_like, optional
        Pullback matrix from ambient to projective variety. If supplied, computes the Euler characteristic 
        on the variety, by default None.
    metric_fn : callable
        Function representing metric tensor in local coordinates.
    cy_dim : int, optional
        Complex dimension of Calabi-Yau.

    Returns
    -------
    chi: array-like
        Contribution to Euler characteristic for each point in `data`. 
    """
    p, weights, dVol_Omega = data
    p = math_utils.to_real(p)
    
    riem = curvature.riemann_tensor_kahler(p, metric_fn, pullbacks)

    if cy_dim == 2:
        _c_2 = chern2(riem)
        c_n = jnp.einsum('...ij,...xy,...ixjy->...', eps_2d, eps_2d, _c_2)
    else:
        c_n = chern3(riem)

    prefactor = 1./math.factorial(cy_dim)
    norm_factor = (-2*1.j)**cy_dim * prefactor  # convert from C^3 to R^6 - convention, since dVol_{CY} = w^n/n!
    chi = norm_factor * weights/dVol_Omega * c_n
    return chi
