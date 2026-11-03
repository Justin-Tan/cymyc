import jax
import jax.numpy as jnp
from jax import grad, jit, jacfwd

import numpy as np
from tqdm import tqdm
from functools import partial
from typing import Callable, List, Sequence
from jaxtyping import Array, Float, Complex, ArrayLike

from . import fubini_study, curvature
from utils import math_utils, poly_utils
from utils import gen_utils as utils


def ddbar_beta(p: Float[Array, "i"], two_form: Callable[[Float[Array, "... i i"]], Array], 
               *args) -> Complex[Array, "... i i i i"]:
    r"""ddbar of (1,1)--form -> (2,2) form.
    Convention: $\partial\bar\partial\beta_{\mu \overline\nu} \rightarrow \beta_{\mu \overline\nu, \rho, \overline\sigma}$
    """
    dim = p.shape[-1] // 2
    real_Hessian = jacfwd(jacfwd(two_form))(p, *args)

    # Decompose Hessian into real, imaginary parts,
    # combine using Wirtinger derivative
    d2_dx2 = real_Hessian[...,:dim,:dim]
    d2_dy2 = real_Hessian[...,dim:,dim:]
    d2_dydx = real_Hessian[...,:dim,dim:]
    d2_dxdy = real_Hessian[...,dim:,:dim]

    four_form = 0.25 * jnp.squeeze(jax.lax.complex(d2_dx2 + d2_dy2, d2_dydx - d2_dxdy))
    four_form -= jnp.einsum("...ijkl->kjil", four_form)
    four_form -= jnp.einsum("...ijkl->ilkj", four_form)
    four_form += jnp.einsum("...ijkl->klij", four_form)
    return four_form

def Tr_End_V_wedge(p: Float[Array, "i"], curvature_fn_alpha: Callable[[Float[Array, "... i i"]], Array],
          curvature_fn_beta: Callable[[Float[Array, "... i i"]], Array] | None = None) -> Complex[Array, "... i i i i"]:
    r"""Wedge and trace of End(V)-valued forms,
    
    Parameters
    ----------
    p : Float[Array, "i"]
    alpha : Callable[[Float[Array, "... i i"]], Array]
        (1,1)-form valued in End(V), $\alpha^s_{t \mu \overline\nu}$
    beta : Callable[[Float[Array, "... i i"]], Array]
        (1,1)-form valued in End(V), $\beta^s_{t \mu \overline\nu}$

    Returns
    -------
    a_wedge_b : Complex[Array, "... i i i i"]
        $ \mathrm{Tr}(\alpha^a_b \wedge \beta^b_d) $
    """
    alpha = curvature_fn_alpha(p)
    if curvature_fn_beta is None:
        beta = alpha
    else:
        beta = curvature_fn_beta(p)
    a_wedge_b = jnp.einsum("...rsuv,...->stwx...rtuvwx", alpha, beta)
    a_wedge_b -= jnp.einsum("...ijkl->kjil", a_wedge_b)
    a_wedge_b -= jnp.einsum("...ijkl->ilkj", a_wedge_b)
    a_wedge_b += jnp.einsum("...ijkl->klij", a_wedge_b)
    return a_wedge_b

def source(p, metric_curvature_fn, curvature_fn):
    Tr_R_sq = Tr_End_V_wedge(p, metric_curvature_fn)
    Tr_F_sq = Tr_End_V_wedge(p, curvature_fn)
    return 0.25 * (Tr_R_sq - Tr_F_sq)

def delta_omega(p, g_inv, beta_fn, kappa):
    _ddbar_beta = ddbar_beta(p, beta_fn)
    return 1./(2 * kappa) * jnp.einsum("...vu, ...uvxy->...xy", g_inv, _ddbar_beta)

def bilinear_integrand(p, g_inv, test_form_fn):
    r"""
        Integrand for the bilinear form $A(t_I, t_J)$from the Galerkin basis.
    """
    ddbar_t = ddbar_beta(p, test_form_fn)  # (..., I, u, v, x, y)
    mu = jnp.einsum("...vu, ...Iuvxy->...Ixy", g_inv, ddbar_t)
    Tr_mu = jnp.einsum("...vu, ...Iuv->...I", g_inv, mu)
    integrand = 0.5 * (jnp.einsum("...I, ...J->...IJ", Tr_mu, Tr_mu)) - jnp.einsum(
        "...vu, ...sr, ...Iuv, ...Jrs->...IJ", g_inv, g_inv, mu, mu)
    return integrand

def gram_integrand(p, g_inv, test_form_fn):
    r"""
        Gram matrix integrand formed from Galerkin basis.
    """
    ddbar_t = ddbar_beta(p, test_form_fn)  # (..., I, u, v, x, y)
    mu = jnp.einsum("...vu, ...Iuvxy->...Ixy", g_inv, ddbar_t)
    Tr_mu = jnp.einsum("...vu, ...Iuv->...I", g_inv, mu)
    G_integrand = jnp.einsum("...vu, ...sr, ...Iuv, ...Jrs->...IJ", g_inv, g_inv, mu, mu) - 0.25 * (
        jnp.einsum("...I, ...J->...IJ", Tr_mu, Tr_mu))
    return G_integrand

def bilinear_and_gram_integrand(p, g_inv, test_form_fn):
    r"""
        Bilinear and Gram matrix integrand formed from Galerkin basis.
    """
    ddbar_t = ddbar_beta(p, test_form_fn)  # (..., I, u, v, x, y)
    mu = jnp.einsum("...vu, ...Iuvxy->...Ixy", g_inv, ddbar_t)
    Tr_mu = jnp.einsum("...vu, ...Iuv->...I", g_inv, mu)
    A_integrand = 0.5 * (jnp.einsum("...I, ...J->...IJ", Tr_mu, Tr_mu)) - jnp.einsum(
        "...vu, ...sr, ...Iuv, ...Jrs->...IJ", g_inv, g_inv, mu, mu)
    G_integrand = jnp.einsum("...vu, ...sr, ...Iuv, ...Jrs->...IJ", g_inv, g_inv, mu, mu) - 0.25 * (
        jnp.einsum("...I, ...J->...IJ", Tr_mu, Tr_mu))
    return A_integrand, G_integrand

def linear_integrand(p, g_inv, rho, test_form_fn):
    r"""
        Linear form integrand $\int_X \rho \wedge t$
    """
    t = test_form_fn(p)  # (..., I, u, v)
    Lambda_rho = jnp.einsum("...vu, ...uvxy->...xy", g_inv, rho)
    Tr_t = jnp.einsum("...vu, ...Iuv->...I", g_inv, t)
    Tr_Lambda_rho = jnp.einsum("...vu, ...uv->...", g_inv, Lambda_rho)
    integrand = 0.5 * Tr_Lambda_rho[..., None] * Tr_t - jnp.einsum(
        "...vu, ...sr, ...uv, ...Irs->...I", g_inv, g_inv, Lambda_rho, t)
    return -integrand

# TODO: should this be registered as a pytree?
class HullStromingerSolver:
    def __init__(self, cy_dim: int, monomials: List[ArrayLike], ambient: np.array, 
                 dQdz_monomials: List[ArrayLike], dQdz_coeffs: List[ArrayLike],
                 metric_fn: Callable, metric_curvature_fn: Callable, curvature_fn: Callable, beta_fn: Callable, 
                 pb_fn: Callable, kappa: float, twisting_degree: int, data_dtype : jnp.dtype = jnp.complex64):

        self.cy_dim = cy_dim
        self.monomials = monomials
        self.ambient = ambient
        self.dQdz_monomials = dQdz_monomials
        self.dQdz_coeffs = dQdz_coeffs
        self.twisting_degree = twisting_degree

        self.metric_fn = metric_fn
        self.metric_curvature_fn = metric_curvature_fn
        self.curvature_fn = curvature_fn
        self.beta_fn = beta_fn
        self.kappa = kappa
        self.pb_fn = pb_fn
        
        self.data_dtype = data_dtype
        self.min_ev_ratio = 10**(-9)

        self.Ok_powers = jnp.asarray(poly_utils.monomial_basis(ambient, self.twisting_degree))
        n_homo_coords = np.array(self.ambient) + 1
        n_asym = n_homo_coords * (n_homo_coords - 1) // 2
        self.n_sections = self.Ok_powers.shape[0] * np.sum(n_asym)

    def test_form(self, p: jnp.ndarray):
        p_c = math_utils.to_complex(p)
        pb = self.pb_fn(p_c)

        test_form_joint = jnp.zeros((self.n_sections, self.cy_dim, self.cy_dim), dtype=self.data_dtype)
        for i in range(len(self.ambient)):
            s, e = int(np.sum(self.ambient[:i]) + i), int(np.sum(self.ambient[:i+1]) + i + 1)
            n_c, as_idx = e-s, jnp.triu_indices(e-s,1)
            p_ambient_i = jax.lax.dynamic_slice(p_c, (s,), (n_c,))

            # Galerkin asis of sections of $\Omega_X(2)$ in ambient P^1 x ... x P^n
            Z_i = p_ambient_i  # coords on the i-th ambient space
            Z_norm_sq_i = jnp.sum(jnp.abs(Z_i)**2)
            pb_i = jax.lax.dynamic_slice(pb, (0,s), (self.cy_dim, n_c))
            Z_dz_i_pb = jnp.einsum('...i, ...jk -> ...ikj', Z_i, pb_i)
            basis_form_i_pb = Z_dz_i_pb - jnp.einsum('...ijv->...jiv', Z_dz_i_pb)
            basis_form_i_pb_v = basis_form_i_pb[as_idx[0],as_idx[1],:]  # (..., i, \bar\nu)
            test_basis_i = jnp.einsum("...iv, ...ju->...ijuv", basis_form_i_pb_v, jnp.conjugate(basis_form_i_pb_v))
            test_basis_i = jnp.reshape(test_basis_i, (*test_basis_i.shape[:-4], -1, *test_basis_i.shape[-2:]))  # (..., i*j, u, v)

            # modulate by position-dependent coeffs
            Ok_monomials_i = poly_utils.monomial_evaluate_log(p, self.Ok_powers)
            coeffs = jnp.einsum("...a, ...b->...ab", Ok_monomials_i, jnp.conj(Ok_monomials_i)) / Z_norm_sq_i**self.twisting_degree
            test_form_i = jnp.einsum("...ab, ...kuv->...abkuv", coeffs, test_basis_i)
            test_form_i = jnp.reshape(test_form_i, (*test_form_i.shape[:-5], -1, *test_form_i.shape[-2:]))  # (..., I, u, v)
            test_form_joint += test_form_i

        return test_form_joint


    def compute_bilinear_form_batched(self, data):
        A = jnp.zeros((self.n_sections, self.n_sections), dtype=self.data_dtype)
        b = jnp.zeros((self.n_sections,), dtype=self.data_dtype)
        G = jnp.zeros_like(A)

        n = 0
        _data_gen = lambda: zip(*data)

        for t, data in enumerate(tqdm(_data_gen(), total=len(data[0]), desc="Computing bilinear form")):
            p, w, pb, *aux = data
            g_inv, rho = aux
            B = p.shape[0]

            _A_integrand, _G_integrand = bilinear_and_gram_integrand(p, g_inv, self.test_form)
            _b_integrand = linear_integrand(p, g_inv, rho, self.test_form)

            _A = jnp.mean(w[..., None, None] * _A_integrand, axis=0)
            _G = jnp.mean(w[..., None, None] * _G_integrand, axis=0)
            _b = jnp.mean(w[..., None] * _b_integrand, axis=0)

            A = math_utils.online_update_array(A, _A, n, B)
            G = math_utils.online_update_array(G, _G, n, B)
            b = math_utils.online_update_array(b, _b, n, B)
            n += B

        return A, G, b

    def solve_system(self, A, G, b):
        # Solve the linear system A w = b in the Galerkin basis
        A = -1./(2 * self.kappa) * A
        eigvals, eigvecs = jnp.linalg.eigh(G)
        norm_eigvals = eigvals / eigvals[-1]
        mask = norm_eigvals > self.min_ev_ratio
        eigvecs = eigvecs[:, mask][mask, :]
        Lambda = jnp.diag(eigvals[mask])
        G = eigvecs @ Lambda @ jnp.conjugate(jnp.transpose(eigvecs))
        T = eigvecs @ jnp.linalg.inv(jnp.sqrt(Lambda))
        A = jnp.conjugate(jnp.transpose(T)) @ A @ T
        b = jnp.conjugate(jnp.transpose(T)) @ b
