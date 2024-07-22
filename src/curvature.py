"""Calculation of various curvature-related quantities for a general Kahler manifold.
Note some functions may specialise to the case of a projective variety.
In general these functions expect local `c_dim` complex coordinates `z` as with the 
real and imaginary parts concatenated to form a real-valued `2*c_dim` vector,
`x = [Re(z); Im(z)]`

"""

import jax
import jax.numpy as jnp
from jax import grad, jit, jacfwd

from jaxtyping import Array, Float, Complex, ArrayLike
from functools import partial
from typing import Callable

# custom
from .utils import math_utils

# def greet(name: str) -> str:
#     """Greet someone.

#     Parameters
#     ----------
#     name
#         The name of the person to greet.

#     Returns
#     -------
#     A greeting message.
#     """
#     return f"Hello {name}!"

def del_z(p: Float[Array, "i"], fun: Callable[[Array], Array], *args) -> Array:
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

def del_bar_z(p: Float[Array, "i"], fun: Callable[[Array], Array], *args) -> Array:
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


@partial(jit, static_argnums=(1,))
def del_z_bar_del_z(p: Float[Array, "i"], fun: Callable[[Array], Array], *args, 
                    wide: bool = False) -> Array:
    r"""Computes ddbar of a given function.
    
    Expects functions with real domain. Computes the full Hessian matrix corresponding to 
    $\bar{\partial} \partial f$
    
    Parameters
    ----------
    p : array_like  
        2*complex_dim real inhomogeneous coords at which `fun` is evaluated. Shape [i].
    fun : callable
        Locally defined function fun: $\mathbb{R}^m -> \mathbb{C}^{a,b,c...}$ sending real-valued 
        inputs to complex-valued outputs
    wide : Flag to use reverse-mode autodiff if function is wide, i.e. if the output of 
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

@partial(jit, static_argnums=(1,))
def christoffel_symbols_kahler(p: Float[Array, "i"], metric_fn: Callable[[Array], Array], 
                               pullbacks: Complex[Array, "i proj_dim"] =None):
    r"""Returns Levi-Civita pullback holomorphic connection on variety $\iota: X \hookrightarrow P^n$.

    Parameters
    ----------
    p : array_like  
        2*complex_dim real inhomogeneous coords at which `fun` is evaluated. Shape [i].
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
        Pullback tensor from ambient to projective variety. If supplied, computes Christoffels
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
def christoffel_symbols_kahler_antiholo(p: Float[Array, "i"], metric_fn: Callable[[Array], Array], 
                               pullbacks: Complex[Array, "i proj_dim"] =None):
    r"""Returns Levi-Civita pullback antiholomorphic connection on variety $\iota: X \hookrightarrow P^n$.
    """
    return jnp.conjugate(christoffel_symbols_kahler(p, metric_fn, pullbacks))

# @partial(jit, static_argnums=(1,3))
# def riemann_tensor_kahler(p, metric_fn, pullbacks=None, return_aux=False):
#     # TODO: VERIFY SYMMETRIES
#     """
#     Arguments
#     ---------
#         'p'         : Point in homogeneous coordinates in ambient space.
#         'metric_fn' : Partially closed function using `jax.tree_utils.Partial`
#                       or `partial` only accepting `p`.
                    
#     Returns
#     -------
#         `riemann`: (1,3) Riemann tensor corresponding to the Kahler 
#                    connection $R^{\kappa}_{\lambda \mu \bar{\nu} $.
#                    See page 335, (8.97) of Nakahara.
    
#     Optionally returns output of `riemann_tensor_kahler_v2` $(R^{\kappa}_{\lambda \bar{\mu} \nu})$.
#     """

#     print('JITTING RIEMANN')
#     del_bar_cs = del_bar_z(p, partial(christoffel_symbols_kahler, metric_fn=metric_fn, 
#         pullbacks=pullbacks))
#     if pullbacks is not None:
#         del_bar_cs = jnp.einsum('...ab, ...ijkb->...ijka', jnp.conjugate(pullbacks), del_bar_cs)
    
#     # $ R^{\kappa}_{\lambda \mu \bar{\nu} $
#     riemann = jnp.einsum('...ijkl->...ikjl', -del_bar_cs)

#     if return_aux is True:
#         # $ R^{\kappa}_{\lambda \bar{\mu} \nu} $
#         riemann2 = jnp.einsum('...ijkl->...iklj', del_bar_cs)
#         return riemann, riemann2

#     return riemann

# @partial(jit, static_argnums=(1,))
# def riemann_tensor_kahler_v2(p, metric_fn, pullbacks=None):
#     # TODO: VERIFY SYMMETRIES
#     """
#     Returns
#     -------
#         `riemann`: (1,3) Riemann tensor corresponding to the Kahler 
#                    connection $ R^{\kappa}_{\lambda \bar{\mu} \nu} $.
#                    See page 329, (8.75a) of Nakahara.
        
#         Related to output of `riemann_tensor_kahler` by interchange of 
#         second-last holo.index and last anti-holo. index. (See below 
#         8.74 in Nakahara).
#     """
    
#     del_bar_cs = del_bar_z(p, partial(christoffel_symbols_kahler, metric_fn=metric_fn, 
#         pullbacks=pullbacks))
#     if pullbacks is not None:
#         del_bar_cs = jnp.einsum('...ab, ...ijkb->...ijka', jnp.conjugate(pullbacks), del_bar_cs)

#     riemann = jnp.einsum('...ijkl->...iklj', del_bar_cs)

#     return riemann


# @partial(jit, static_argnums=(1,))
# def ricci_tensor_kahler(p, metric_fn, pullbacks=None):
#     # TODO: VERIFY SYMMETRIES
#     """
#     Returns
#     -------
#         `ricci`: (0,2) Ricci tensor corresponding to the Kahler 
#                  connection $ R_{\mu \bar{\nu}}$.
#     """

#     riemann = riemann_tensor_kahler(p, metric_fn, pullbacks)
#     # Contraction $ R^{\kappa}_{\lambda \kappa \bar{\nu} $
#     ricci_tensor = jnp.einsum('...kikj->...ij', riemann)
    
#     return ricci_tensor

# @partial(jit, static_argnums=(1,))
# def ricci_form_kahler(p, metric_fn, pullbacks=None):
#     """
#     Componentwise, :math: \rho_{\mu\bar{\nu}} = i R_{\mu \bar{\nu}}.
#     """
#     ricci_form = -1.j * del_z_bar_del_z(p, partial(math_utils.log_det_fn, g=metric_fn))

#     if pullbacks is not None:
#         ricci_form = jnp.einsum('...ia,...ab,...jb->...ij', pullbacks, jnp.squeeze(ricci_form),
#             jnp.conjugate(pullbacks))

#     return ricci_form

# @partial(jit, static_argnums=(1,))
# def ricci_scalar(p, metric_fn, pullbacks=None):
#     """
#     Parameters
#     ----------
#         `p` : Coords at which R is eval'd. Shape [..., i]
#     Returns
#     -------
#         `R` : Ricci scalar, $R = g^{\mu \bar{\nu}}R_{\mu \bar{\nu}}$. Shape [...]
#     """
#     g_herm = metric_fn(p)
#     ricci_tensor = ricci_tensor_kahler(p, metric_fn, pullbacks)
#     g_inv = jnp.linalg.inv(g_herm)

#     R = jnp.einsum('...ij, ...ji->...', g_inv, ricci_tensor)
#     return jnp.real(R)

# @partial(jit, static_argnums=(1,))
# def ricci_scalar_from_form(p, metric_fn, pullbacks=None):
#     """
#     Parameters
#     ----------
#         `p` : Coords at which R is eval'd. Shape [..., i]
#     Returns
#     -------
#         `R` : Ricci scalar, $R = g^{\mu \bar{\nu}}R_{\mu \bar{\nu}}$. Shape [...]
#     """
#     g_herm = metric_fn(p)
#     ricci_form = ricci_form_kahler(p, metric_fn, pullbacks)  # Ricci form in ambient space
#     ricci_tensor = -1.j * ricci_form

#     g_inv = jnp.linalg.inv(g_herm)
#     R = jnp.einsum('...ij, ...ji->...', g_inv, ricci_tensor)
#     return jnp.real(R)

# @partial(jit, static_argnums=(1,))
# def jacobian_fn(p, fun, *args, wide=False):
#     """
#     Jacobian fn for functions with complex arguments.
#     Parameters
#     ----------
#         `p`            : 2*complex_dim real inhomogeneous coords at 
#                          which `fun` is evaluated. Shape [i].
#         `fun`          : fun: R^m -> C^{a,b,c...} Real-valued inputs, complex-
#                          valued outputs
#     Returns
#     ----------
#         (`dfun_dz`, `dfun_dz_bar`)      : Holomorphic + (anti) derivatives.
#     """
    
#     if wide is True:
#         grad_op = jax.grad
#     else:
#         grad_op = jax.jacfwd

#     dim = p.shape[-1]//2  # complex dimension

#     real_Jac_fun_p = grad_op(fun)(math_utils.to_complex(p), *args)
#     dfun_dx = real_Jac_fun_p[..., :dim]
#     dfun_dy = real_Jac_fun_p[..., dim:]

#     dfun_dz = 0.5 * (dfun_dx - 1.j * dfun_dy)
#     dfun_dz_bar = 0.5 * (dfun_dx + 1.j * dfun_dy)
    
#     return jnp.squeeze(dfun_dz), jnp.squeeze(dfun_dz_bar)
