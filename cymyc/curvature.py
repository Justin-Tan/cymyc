r"""Calculation of various curvature-related quantities for a general Kahler manifold $X$. This module may also be used
to compute curvature information on a real manifold without complex structure, by omitting the imaginary part
of the local coordinates for $p \in X$.

Note the following:

* Curvature quantities involve $n$-th order derivatives of some generic function `fun`. These are 
computed with autodiff - a good usage pattern is to make a partial closure to bind all arguments to `fun`
except the coordinate dependence.
* These functions expect local coordinates `z` in a `c_dim`-dimensional space, with the 
real and imaginary parts concatenated to form a real-valued `2*c_dim` vector, `p = [Re(z); Im(z)]`.

!!! info
    When transforming functions which compute derivatives of some `fun` passed as an input, you will need to either:

    * In the case of `jit`, specify that `fun` is a static argument.
    * Wrap `fun` in a `jax.tree_util.Partial` closure to make it compatible with Jax transformations.

    See the below example.

!!! example
    ``` py
    import jax
    import jax.numpy as jnp
    fun = lambda x: jnp.sum(jnp.cos(x))

    >>> jax.jit(del_z, static_argnums=(1,))(p, fun)  # ok
    >>> jax.jit(del_z)(p, jax.tree_util.Partial(fun))  # ok
    >>> jax.jit(del_z)(p, fun)  # TypeError
    ```
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, jacfwd

from functools import partial
from typing import Callable, Sequence
from jaxtyping import Array, Float, Complex, ArrayLike

# custom
from .utils import math_utils

def del_z(p: Float[Array, "i"], fun: Callable[[Float[Array, "..."]], Array], *args) -> Complex[Array, "... i"]:
    r"""Holomorphic derivative of a function.

    Parameters
    ----------
    p : array_like
        2 * `complex_dim` real coords with `float` type at which `fun` is evaluated. 
        Consists of the concatenation of real and imaginary parts along the last axis.
    fun : callable
        Locally defined function fun: $\mathbb{R}^m -> \mathbb{C}^{a,b,c...}$ sending real-valued 
        inputs to complex-valued outputs
    Returns
    -------
    dfun_dz :   array_like
        Holomorphic derivative of `fun`.

    Notes
    -----
    Computes holomorphic Wirtinger derivative, w.r.t. complex $p = x + iy$.

    $$
    \frac{\partial f}{\partial z} = \frac{1}{2}\left( \frac{\partial f}{\partial x} - i \frac{\partial f}{\partial y} \right)
    $$
    
    Examples
    --------
    >>> p = jnp.ones((8,))
    >>> fun = lambda x: jnp.sum(jnp.cos(x))
    >>> del_z(p, fun)
    Array([-0.42073548+0.42073548j, -0.42073548+0.42073548j,
           -0.42073548+0.42073548j, -0.42073548+0.42073548j], dtype=complex64)
    """
    
    dim = p.shape[-1]//2  # complex dimension
    real_Jac_fun_p = jacfwd(fun)(p, *args)
    dfun_dx = real_Jac_fun_p[..., 0:dim]
    dfun_dy = real_Jac_fun_p[..., dim:]
    dfun_dz = 0.5 * (dfun_dx - 1.j * dfun_dy)
    
    return jnp.squeeze(dfun_dz)

def del_bar_z(p: Float[Array, "i"], fun: Callable[[Float[Array, "..."]], Array], *args) -> Complex[Array, "... i"]:
    r"""Anti-holomorphic derivative of a function.

    Parameters
    ----------
    p : array_like
        2 * `complex_dim` real coords at 
        which `fun` is evaluated. Shape [i].
    fun : callable
        Locally defined function fun: $\mathbb{R}^m -> \mathbb{C}^{a,b,c...}$ sending real-valued 
        inputs to complex-valued outputs
    Returns
    -------
    dfun_dz_bar   :   array_like
        Anti-holomorphic derivative of `fun`.
    """
    
    dim = p.shape[-1]//2  # complex dimension
    real_Jac_fun_p = jacfwd(fun)(p, *args)
    dfun_dx = real_Jac_fun_p[..., 0:dim]
    dfun_dy = real_Jac_fun_p[..., dim:]
    dfun_dz_bar = 0.5 * (dfun_dx + 1.j * dfun_dy)
    
    return jnp.squeeze(dfun_dz_bar)


def del_z_bar_del_z(p: Float[Array, "i"], fun: Callable[[Float[Array, "..."]], Array], *args, 
                    wide: bool = False) -> Complex[Array, "... i i"]:
    r"""Computes the composition $\partial \circ \overline{\partial}$ of a given function. Note this 
    coincides with the Hessian $\nabla df$ on a complex manifold.
    
    Parameters
    ----------
    p : array_like  
        2 * `complex_dim` real coords at which `fun` is evaluated. Shape [i].
    fun : callable
        Locally defined function fun: $\mathbb{R}^m -> \mathbb{C}^{a,b,c...}$ sending real-valued 
        inputs to complex-valued outputs
    wide : bool
        Flag to use reverse-mode autodiff if function is wide, i.e. if the output of 
        `fun` is a scalar. 
    Returns
    -------
    dfun_dz_bar_dz : array_like
        $\bar{\partial} \partial f$. Shape [..., \mu, \bar{\nu}]. Note holomorphic index 
        comes first.
    """
    
    if wide is True:
        inner_grad = jax.grad
    else:
        inner_grad = jax.jacfwd
    
    dim = p.shape[-1]//2  # complex dimension
    real_Hessian = jacfwd(inner_grad(fun))(p, *args)
    
    # Decompose Hessian into real, imaginary parts,
    # combine using Wirtinger derivative
    d2f_dx2 = real_Hessian[...,:dim,:dim]
    d2f_dy2 = real_Hessian[...,dim:,dim:]
    d2f_dydx = real_Hessian[...,:dim,dim:]
    d2f_dxdy = real_Hessian[...,dim:,:dim]

    return 0.25 * jnp.squeeze(jax.lax.complex(d2f_dx2 + d2f_dy2, d2f_dydx - d2f_dxdy))


def christoffel_symbols_kahler(p: Float[Array, "i"], metric_fn: Callable[[Array], Array], 
                               pullbacks: Complex[Array, "cy_dim i"] = None) -> Complex[Array, "j j j"]:
    r"""Returns holomorphic Levi-Civita connection coefficients, with optional 
    support for variety $\iota: X \hookrightarrow \mathbb{P}^n$.
    Schematically, if $g$ is the metric tensor,
    $$ \Gamma \sim g^{-1} \cdot \partial g~. $$
    Parameters
    ----------
    p : array_like  
        2 * `complex_dim` real coords at which `fun` is evaluated. Shape [i].
    metric_fn : callable
        Function representing metric tensor in local coordinates $g : \mathbb{R}^m -> \mathbb{C}^{a,b...}$.
    Returns
    -------
    gamma_holo: array_like
        Holomorphic Christoffel symbols of the Kahler metric.
        $\Gamma^{\lambda}_{\mu \nu}$. Shape [...,k,i,j],
        symmetric in (i,j). The Kahler conditions 
        imply $\Gamma^{\lambda}_{\mu \nu}$ and its
        conjugate are the only nonzero connection coeffs.
    Other Parameters
    ----------------
    pullbacks : array_like, optional
        Pulllback matrices from ambient to projective variety. If supplied, computes Christoffels
        on the variety.
    """
    
    # metric_fn(p) gives g_{\mu \bar{\nu}}
    g_inv = jnp.linalg.inv(metric_fn(p))  # g^{\bar{\nu} \mu}
    jac_g_holo = del_z(p, metric_fn)
    
    if pullbacks is not None:
        jac_g_holo = jnp.einsum('...ab, ...ijb->...ija', pullbacks, jac_g_holo)
    
    gamma_holo = jnp.einsum('...kl, ...jki->...lij', g_inv, jac_g_holo)
    return gamma_holo 

@partial(jit, static_argnums=(1,))
def _christoffel_symbols_kahler_antiholo(p: Float[Array, "i"], metric_fn: Callable[[Array], Array], 
                               pullbacks: Complex[Array, "cy_dim i"] = None) -> Complex[Array, "j j j"]:
    r"""Returns Levi-Civita pullback antiholomorphic connection, with support on variety 
    $\iota: X \hookrightarrow P^n$.
    """
    return jnp.conjugate(christoffel_symbols_kahler(p, metric_fn, pullbacks))

@partial(jit, static_argnums=(3,))
def riemann_tensor_kahler(p: Float[Array, "i"], metric_fn: Callable[[Array], Array], 
                          pullbacks: Complex[Array, "cy_dim i"] = None,
                          return_aux: bool = False) -> Complex[Array, "j j j j"] | Sequence[Array]:
    # TODO: VERIFY SYMMETRIES
    r"""Returns Riemann curvature tensor on a Kähler manifold, with optional support for variety $\iota: X \hookrightarrow \mathbb{P}^n$.
    This can also be used to construct the Riemann tensor on a real manifold by omission of the imaginary part of `p`. Schematically,
    $$
    \textsf{Riem} \sim \partial \Gamma + \Gamma \Gamma ~.
    $$
    Parameters
    ----------
    p : array_like  
        2 * `complex_dim` real coords at which `fun` is evaluated. Shape [i].
    metric_fn : callable
        Function representing metric tensor in local coordinates $g : \mathbb{R}^m -> \mathbb{C}^{a,b...}$.
                    
        !!! warning
            This function explicitly instantiates the complex Hessian of `metric_fn` - this may result in 
            memory issues if `vmap`-ing over a large batch. Try reducing the batch size or reducing the complexity
            of `metric_fn` if memory-constrained.
    Returns
    -------
    riemann: array_like
        (1,3) Riemann tensor corresponding to the Kahler connection $R^{\kappa}_{\lambda \mu \overline{\nu}}$.
        See page 335, (8.97) of Nakahara.
    Other Parameters
    ----------------
    pullbacks : array_like, optional
        Pulllback matrices from ambient to projective variety. If supplied, computes Riemann tensor
        on the variety.
    """

    del_bar_cs = del_bar_z(p, partial(christoffel_symbols_kahler, metric_fn=metric_fn, 
        pullbacks=pullbacks))
    if pullbacks is not None:
        del_bar_cs = jnp.einsum('...ab, ...ijkb->...ijka', jnp.conjugate(pullbacks), del_bar_cs)
    
    # $ R^{\kappa}_{\lambda \mu \bar{\nu} $
    riemann = jnp.einsum('...ijkl->...ikjl', -del_bar_cs)

    if return_aux is True:
        # $ R^{\kappa}_{\lambda \bar{\mu} \nu} $
        riemann2 = jnp.einsum('...ijkl->...iklj', del_bar_cs)
        return riemann, riemann2

    return riemann

@partial(jit, static_argnums=(1,))
def ricci_tensor_kahler(p: Float[Array, "i"], metric_fn: Callable[[Array], Array], 
                        pullbacks: Complex[Array, "cy_dim i"] = None) -> Complex[Array, "j j"]:
    # TODO: VERIFY SYMMETRIES
    r"""Returns Ricci curvature tensor on a Kähler manifold, with optional support for variety 
    $\iota: X \hookrightarrow \mathbb{P}^n$.
    Parameters
    ----------
    p : array_like  
        2 * `complex_dim` real coords at which `fun` is evaluated. Shape [i].
    metric_fn : callable
        Function representing metric tensor in local coordinates $g : \mathbb{R}^m -> \mathbb{C}^{a,b...}$.
    Returns
    -------
    ricci: array_like
        (0,2) Ricci tensor corresponding to the Kahler connection $ R_{\mu \bar{\nu}}$.
    Other Parameters
    ----------------
    pullbacks : array_like, optional
        Pulllback matrices from ambient to projective variety. If supplied, computes Ricci tensor
        on the variety.
    See Also
    --------
    `ricci_form_kahler` : Computes Ricci form as $\partial_{\mu} \overline{\partial}_{\overline{\nu}} \log g$.
    """

    riemann = riemann_tensor_kahler(p, metric_fn, pullbacks)
    # Contraction R^{\kappa}_{\lambda \kappa \bar{\nu}
    ricci_tensor = jnp.einsum('...kikj->...ij', riemann)
    
    return ricci_tensor

@jit
def ricci_form_kahler(p: Float[Array, "i"], metric_fn: Callable[[Array], Array], 
                      pullbacks: Complex[Array, "cy_dim i"] = None) -> Complex[Array, "j j"]:
    r""" Returns Ricci form $\rho$ on a Kähler manifold, with support for variety $\iota: X \hookrightarrow \mathbb{P}^n$. 
    Componentwise, $\rho_{\mu\bar{\nu}} = i R_{\mu \bar{\nu}}$, and
    $$ \rho = \partial \overline{\partial} \log \det g~. $$
    Parameters
    ----------
    p : array_like  
        2 * `complex_dim` real coords at which `fun` is evaluated. Shape [i].
    metric_fn : callable
        Function representing metric tensor in local coordinates $g : \mathbb{R}^m -> \mathbb{C}^{a,b...}$.
    Returns
    -------
    ricci: array_like
        (1,1) Ricci form corresponding to the Kahler connection $ R_{\mu \bar{\nu}}$.
    Other Parameters
    ----------------
    pullbacks : array_like, optional
        Pulllback matrices from ambient to projective variety. If supplied, computes Ricci tensor
        on the variety.
    """
    ricci_form = -1.j * del_z_bar_del_z(p, partial(math_utils.log_det_fn, g=metric_fn))

    if pullbacks is not None:
        ricci_form = jnp.einsum('...ia,...ab,...jb->...ij', pullbacks, jnp.squeeze(ricci_form),
            jnp.conjugate(pullbacks))

    return ricci_form

@partial(jit, static_argnums=(1,))
def ricci_scalar(p: Float[Array, "i"], metric_fn: Callable[[Array], Array], 
                 pullbacks: Complex[Array, "cy_dim i"] = None) -> Complex[Array, ""]:
    r""" Returns Ricci scalar on a Kähler manifold, with support for variety $\iota: X \hookrightarrow \mathbb{P}^n$. 
    Parameters
    ----------
    p : array_like  
        2 * `complex_dim` real coords at which `fun` is evaluated. Shape [i].
    metric_fn : callable
        Function representing metric tensor in local coordinates $g : \mathbb{R}^m -> \mathbb{C}^{a,b...}$.
    Returns
    -------
    R: array_like
        Ricci scalar, $R = g^{\mu \bar{\nu}}R_{\mu \bar{\nu}}$. Shape [...]
    Other Parameters
    ----------------
    pullbacks : array_like, optional
        Pulllback matrices from ambient to projective variety. If supplied, computes Ricci tensor
        on the variety.
    """
    g_herm = metric_fn(p)
    ricci_tensor = ricci_tensor_kahler(p, metric_fn, pullbacks)
    g_inv = jnp.linalg.inv(g_herm)

    R = jnp.einsum('...ij, ...ji->...', g_inv, ricci_tensor)
    return jnp.real(R)

@partial(jit, static_argnums=(1,))
def ricci_scalar_from_form(p: Float[Array, "i"], metric_fn: Callable[[Array], Array], 
                           pullbacks: Complex[Array, "cy_dim i"] = None) -> Complex[Array, ""]:
    g_herm = metric_fn(p)
    ricci_form = ricci_form_kahler(p, metric_fn, pullbacks)  # Ricci form in ambient space
    ricci_tensor = -1.j * ricci_form

    g_inv = jnp.linalg.inv(g_herm)
    R = jnp.einsum('...ij, ...ji->...', g_inv, ricci_tensor)
    return jnp.real(R)

