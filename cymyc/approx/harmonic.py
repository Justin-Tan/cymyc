r"""
Approximations of harmonic one-forms on Calabi-Yau manifolds, i.e. forms $\eta$ satisfying $\Delta_g \eta = 0$. 
"""

import jax
import numpy as np  # original CPU-backed NumPy
import jax.numpy as jnp

from jax import random
from jax import grad, jit, vmap, jacfwd

from typing import List, Callable, Mapping, Tuple
from jaxtyping import Array, Float, Complex, ArrayLike

from functools import partial
from flax import linen as nn

# custom
from . import models, measures
from .. import alg_geo, curvature, fubini_study

from ..moduli import wp
from ..utils import math_utils
from ..utils import gen_utils as utils

# TODO Register as pytree
# This will be fixed soon, but be careful not to mutate the class after initialization. This will 
# either trigger recompilation, or result in undefined behaviour!
class Harmonic(wp.WP):
    r"""
    Approximation of harmonic one-forms for CYs with single complex structure modulus,
    i.e. $h^{2,1} = 1$. 
    """
    def __init__(self, cy_dim: int, monomials: List[np.array], ambient: np.array, 
                 deformations: List[Callable], dQdz_monomials: List[np.array], 
                 dQdz_coeffs: List[np.array], metric_fn: Callable,
                 pb_fn: Callable, coeff_fn: Callable, psi: float):
        
        super().__init__(cy_dim, monomials, ambient)
        jax.config.update("jax_enable_x64", False)
        self.h_21 = 1
        self.deformation = deformations[0]
        self.dQdz_monomials = dQdz_monomials
        self.dQdz_coeffs = dQdz_coeffs
        self.metric_fn = metric_fn
        self.pb_fn = pb_fn
        self.coeff_fn = coeff_fn
        self.psi = psi
        self.proj_idx = jnp.asarray(utils._generate_proj_indices(self.degrees))
        self.bounds = jnp.cumsum(jnp.concatenate((jnp.zeros(1), self.degrees)))
        self.n_transitions = utils._patch_transitions(self.n_hyper, len(self.ambient), self.degrees)
        self.n_ambient = len(self.ambient)
        self.toggle_transition_loss = False

        if (self.n_hyper > 1) or (len(self.ambient) > 1):
            self.fs_metric_fn = partial(fubini_study._fubini_study_metric_homo_gen_pb_cicy,
                        dQdz_monomials=self.dQdz_monomials, dQdz_coeffs=self.dQdz_coeffs,
                        n_hyper=self.n_hyper, cy_dim=self.cy_dim, n_coords=self.n_homo_coords,
                        ambient=tuple(self.ambient), k_moduli=None, ambient_out=False, cdtype=np.complex64)
            self._pb_fn_set_dQ_idx = partial(alg_geo._pullbacks_cicy_set_dQ_elim,
                        dQdz_monomials=self.dQdz_monomials, dQdz_coeffs=self.dQdz_coeffs,
                        n_hyper=self.n_hyper, cy_dim=self.cy_dim, n_coords=self.n_homo_coords,
                        aux=False, cdtype=np.complex64)
        else:
            self.fs_metric_fn = partial(fubini_study.fubini_study_metric_homo_pb,
                        dQdz_info=(self.dQdz_monomials, self.dQdz_coeffs),
                        cy_dim=self.cy_dim, ambient_out=False, cdtype=np.complex64)

    def _zeta(self, x):

        p = math_utils.to_complex(x)

        g_FS_inv = jnp.conjugate(self._fs_metric_inverse_total(p)) # g^{\mu \bar{\nu}}
        dQdz = self._compute_dQdz_inhomo(p, self.dQdz_monomials, self.dQdz_coeffs)  # in the full ambient space
        H = jnp.einsum('...ij, ...ia, ...jb->...ab', g_FS_inv, dQdz, jnp.conjugate(dQdz))
        cm_def = self.deformation(p)
        diffeo = -jnp.einsum('...ab,...uv,...va,...b->...u', jnp.linalg.inv(H), g_FS_inv, 
                             jnp.conjugate(dQdz), cm_def)

        return diffeo

    def _del_bar_zeta(self, x):
        """
        Outputs vector field in ambient space corresponding to diffeomorphism between
        fibres of Kuranishi family. 
        """
        dim = x.shape[-1] // 2  # complex dimension
        real_jac_x = jacfwd(self._zeta)(x)
        dzeta_dx = real_jac_x[..., :dim]
        dzeta_dy = real_jac_x[..., dim:]
        dzeta_dzbar = 0.5 * (dzeta_dx + 1.j * dzeta_dy)

        return dzeta_dzbar

    def form_correction_fn(self, p, params, spectral=False, activation=nn.gelu):
        spectral = False
        n_out =  self.cy_dim * self.h_21 * 2
        vec = models.phi_head(p, params, n_hyper=self.n_hyper, ambient=tuple(self.ambient), 
                              n_out=n_out, spectral=spectral, activation=activation)
        form_correction = jnp.reshape(math_utils.to_complex(vec), (self.cy_dim, self.h_21))

        return form_correction  # don't squeeze


    def section_network(self, p, params, activation=nn.gelu):

        p_c = math_utils.to_complex(p)
        pb = self.pb_fn(p_c)
        g_inv = jnp.linalg.inv(self.fs_metric_fn(p))  # g^{\bar{\nu} \mu}

        n_homo_coords_i = np.array(self.ambient) + 1  # coords for each ambient space factor 
        param_counts = [(n_c*(n_c-1)//2, (n_c+1)*n_c//2) for n_c in n_homo_coords_i]
        n_asym, n_sym = param_counts[-1]

        # (h_{21}, n_asym, n_sym) * n_A if all ambient space factors identical
        coeffs = models.coeff_head(p, params, self.n_homo_coords, tuple(self.ambient), self.h_21, activation)
        T_X_section = jnp.empty((self.h_21, self.cy_dim,), dtype=np.complex64)
        for i in range(len(self.ambient)):
            s, e = int(np.sum(self.ambient[:i]) + i), int(np.sum(self.ambient[:i+1]) + i + 1)
            n_c, as_idx = e-s, jnp.triu_indices(e-s,1)
            p_ambient_i = jax.lax.dynamic_slice(p_c, (s,), (n_c,))

            # Basis of sections in ambient P^1 x ... x P^n
            Z_i = p_ambient_i  # coords on the i-th ambient space
            Z_norm_sq_i = jnp.sum(jnp.abs(Z_i)**2)
            pb_i = jax.lax.dynamic_slice(pb, (0,s), (self.cy_dim, n_c))
            Z_dz_i_pb = jnp.einsum('...i, ...jk -> ...ikj', Z_i, pb_i)
            basis_form_i_pb = Z_dz_i_pb - jnp.einsum('...ijv->...jiv', Z_dz_i_pb)
            basis_form_i_pb_v = basis_form_i_pb[as_idx[0],as_idx[1],:]
            
            O2_s_i = jnp.outer(Z_i,Z_i)
            O2_s_i_v = O2_s_i[jnp.triu_indices(n_c)] / Z_norm_sq_i**2

            T_X_section += jnp.einsum('...hab, ...av, ...vu, ...b->...hu', coeffs[i], jnp.conjugate(basis_form_i_pb_v), 
                    g_inv, O2_s_i_v)

        return T_X_section  # don't squeeze

    def section_network_transformed(self, p, elim_idx, params, activation=nn.gelu):

        p_c = math_utils.to_complex(p)
        pb = self._pb_fn_set_dQ_idx(p_c, elim_idx)
        fs_pb = fubini_study.fubini_study_metric_homo_pb_cicy(p, pb, self.n_homo_coords, tuple(self.ambient))
        g_inv = jnp.linalg.inv(fs_pb) # g^{\bar{\nu} \mu}

        n_homo_coords_i = np.array(self.ambient) + 1  # coords for each ambient space factor 
        param_counts = [(n_c*(n_c-1)//2, (n_c+1)*n_c//2) for n_c in n_homo_coords_i]
        n_asym, n_sym = param_counts[-1]
        basis_forms_pb = jnp.empty((self.n_ambient * n_asym, self.cy_dim), dtype=np.complex64)
        O2_s = jnp.empty((self.n_ambient * n_sym,), dtype=np.complex64)

        # (h_{21}, n_asym, n_sym) * n_A
        coeffs = models.coeff_head(p, params, self.n_homo_coords, tuple(self.ambient), self.h_21, activation)
        T_X_section = jnp.empty((self.h_21, self.cy_dim,), dtype=np.complex64)
        for i in range(len(self.ambient)):
            s, e = int(np.sum(self.ambient[:i]) + i), int(np.sum(self.ambient[:i+1]) + i + 1)
            n_c, as_idx = e-s, jnp.triu_indices(e-s,1)
            p_ambient_i = jax.lax.dynamic_slice(p_c, (s,), (n_c,))

            # Basis of sections in ambient P^1 x ... x P^n
            Z_i = p_ambient_i  # coords on the i-th ambient space
            Z_norm_sq_i = jnp.sum(jnp.abs(Z_i)**2)
            pb_i = jax.lax.dynamic_slice(pb, (0,s), (self.cy_dim, n_c))
            Z_dz_i_pb = jnp.einsum('...i, ...jk -> ...ikj', Z_i, pb_i)
            basis_form_i_pb = Z_dz_i_pb - jnp.einsum('...ijv->...jiv', Z_dz_i_pb)
            basis_form_i_pb_v = basis_form_i_pb[as_idx[0],as_idx[1],:]
            print(f'{self.section_network_transformed.__qualname__}, basis_form shape', basis_form_i_pb.shape)
            
            O2_s_i = jnp.outer(Z_i,Z_i)
            O2_s_i_v = O2_s_i[jnp.triu_indices(n_c)] / Z_norm_sq_i**2
            print(f'{self.section_network_transformed.__qualname__}, O2 shape', O2_s_i.shape)

            T_X_section += jnp.einsum('...hab, ...av, ...vu, ...b->...hu', coeffs[i], jnp.conjugate(basis_form_i_pb_v), 
                    g_inv, O2_s_i_v)

        return T_X_section

    def det_g_fn(self, p):
        g_pred = self.metric_fn(p)
        return jnp.real(jnp.linalg.det(g_pred))

    @staticmethod
    def project_coords(A, mask, size):
        return jnp.squeeze(A[jnp.nonzero(mask, size=size), ...])

    @staticmethod
    def del_z_del_z_bar(p, fun, *args):

        dim = p.shape[-1]//2  # complex dimension
        real_Hessian = jax.jacfwd(jax.jacfwd(fun))(p, *args)

        # Decompose Hessian into real, imaginary parts,
        # combine using Wirtinger derivative
        d2f_dx2 = real_Hessian[...,:dim,:dim]
        d2f_dy2 = real_Hessian[...,dim:,dim:]
        d2f_dydx = real_Hessian[...,:dim,dim:]
        d2f_dxdy = real_Hessian[...,dim:,:dim]

        ddbar_f = 0.25 * jnp.squeeze((d2f_dx2 + d2f_dy2) +  1.j * (d2f_dxdy - d2f_dydx))
        return ddbar_f

    def compute_masks(self, p):
        p_c = math_utils.to_complex(p)
        ones_mask = jnp.logical_not(jnp.isclose(p_c, jax.lax.complex(1.,0.)))
        dQdz_homo = alg_geo.evaluate_dQdz(p_c, self.dQdz_monomials, self.dQdz_coeffs)
        _, good_coord_mask = alg_geo.argmax_dQdz_cicy(p_c, dQdz_homo, self.n_hyper, self.n_homo_coords, True)
        good_coord_mask = good_coord_mask[jnp.nonzero(ones_mask, size=self.n_inhomo_coords)]# g^{\bar{\nu} \mu}
        return ones_mask, good_coord_mask

    @staticmethod
    def polarisation_preserving_property(eta, g_pred):
        g_inv = jnp.linalg.inv(g_pred)  # g^{\bar{\nu} \mu}
        harmonic_contraction = jnp.einsum('...av,...bu,...ab->...uv', g_pred, g_inv, eta)
        return harmonic_contraction - eta

    @staticmethod
    def symmetry_property(eta, g_pred):
        eta_ab = jnp.einsum('...ia, ...ib->...ab', g_pred, eta)
        return eta_ab - jnp.einsum('...ab->...ba', eta_ab)

    @partial(jit, static_argnums=(0,))
    def del_z_harmonic_rep(self, p, pb, good_coord_mask, params):

        ddbar_zeta = self.del_z_del_z_bar(p, self.zeta_complete)
        ddbar_theta = self.del_z_del_z_bar(p, self.section_network, params)

        if self.h_21 == 1:
            d_ref_ambient = jnp.expand_dims(self.project_coords(ddbar_zeta, good_coord_mask, self.cy_dim), axis=0)
        else:
            d_ref_ambient = vmap(self.project_coords, in_axes=(0,None,None))(ddbar_zeta, good_coord_mask, self.cy_dim)
        
        # don't squeeze either
        d_eta_ambient = d_ref_ambient + ddbar_theta
        d_eta = jnp.einsum('...habc, ...ib, ...jc -> ...haij', d_eta_ambient, jnp.conjugate(pb), pb)
        return d_eta

    def harmonic_rep_ambient(self, p, params):
        xi_ambient = self._del_bar_zeta(p)  # [..., n_inhomo_coords, n_homo_coords]
        correction_fibre = curvature.del_bar_z(p, self.section_network, params)  # [..., cy_dim, n_homo_coords]
        return xi_ambient, correction_fibre

    def harmonic_rep(self, p, params):
        r"""
        Constructs harmonic representative by $\bar{\partial}$-exact correction to
        element from the $H_{\bar{\partial}}^{0,1}$ Dolbeault cohomology $\xi$, viz.:
        $ \eta = \xi + \bar{\partial} \theta $.
        """

        p_c = math_utils.to_complex(p)
        pb = self.pb_fn(p_c)
        ones_mask, good_coord_mask = self.compute_masks(p)

        xi_ambient, correction_fibre = self.harmonic_rep_ambient(p, params)
        xi_fibre = vmap(self.project_coords, in_axes=(0,None,None))(xi_ambient, good_coord_mask, self.cy_dim)  # don't squeeze
        # don't squeeze
        # [...,h_{21}, cy_dim, n_homo_coords]
        eta = jnp.einsum('...hai, ...bi->...hab', xi_fibre + correction_fibre, jnp.conjugate(pb))
        
        return eta

    def harmonic_rep_derivative(self, p, params):
        p_c = math_utils.to_complex(p)
        pb = self.pb_fn(p_c)
        ones_mask, good_coord_mask = self.compute_masks(p)

        d_eta_ambient = curvature.del_z(p, self._harmonic_rep_ambient, ones_mask, params)
        d_eta_ambient = jnp.squeeze(self.project_coords(d_eta_ambient, good_coord_mask, self.cy_dim))
        d_eta = jnp.squeeze(jnp.einsum('...abc, ...ib, ...jc -> ...aij', d_eta_ambient, jnp.conjugate(pb), pb))
        return d_eta

    @partial(jit, static_argnums=(0,))
    def harmonic_rep_breakdown(self, p, params):

        p_c = math_utils.to_complex(p)
        pb = self.pb_fn(p_c)
        pb_conj = jnp.conjugate(pb)
        ones_mask, good_coord_mask = self.compute_masks(p)

        xi_ambient, correction_fibre = self.harmonic_rep_ambient(p, params)

        form_ref = jnp.squeeze(self.project_coords(xi_ambient, good_coord_mask, self.cy_dim))
        form_ref = jnp.einsum('...ai,...bi->...ab', form_ref, pb_conj)
        form_correction = jnp.einsum('...ai,...bi->...ab', correction_fibre, pb_conj)

        eta = form_ref + form_correction
        return eta, form_ref, form_correction


    @partial(jit, static_argnums=(0,))
    def codifferential_eta(self, p, pullbacks, g_pred, params):
        r"""
        Finds codifferential of $\alpha \in H^{(0,1)}_{\bar{\partial}}(X; T_X)$.
        This is a smooth section of the holomorphic tangent bundle on X.
        """
        _, good_coord_mask = self.compute_masks(p)
        eta = self.harmonic_rep(p, params)  # [a, \bar{\nu}]
        
        g_inv = jnp.linalg.inv(g_pred)  # g^{\bar{\nu} \mu}
        Gamma_holo = curvature.christoffel_symbols_kahler(p, self.metric_fn, pullbacks)  # [a, \kappa, b]

        # two derivatives of NN
        del_z_eta = self.del_z_harmonic_rep(p, pullbacks, good_coord_mask, params) # [a, \bar{\nu}, \kappa]
        _cov2 = jnp.einsum('...akb, ...bv -> ...avk', Gamma_holo, eta)   # [a, \bar{\nu}, \kappa]
        covariant_derivative_eta = del_z_eta + _cov2
        codiff = jnp.einsum('...vu, ...avu->...a', g_inv, covariant_derivative_eta)

        return jnp.squeeze(codiff)  # section of tangent bundle, shape [..., cy_dim]


    def objective_function(self, data, params, norm_order=1., C=10**0, full_contraction=False):
        r"""Enforces condition $\bar{\partial}^{\dagger} \eta (\in C^{\infty}(X)) = 0$.
        """
        print(f'Compiling {self.objective_function.__qualname__}')

        p, weights, dVol_Omega = data
        pullbacks = vmap(self.pb_fn)(math_utils.to_complex(p))
        g_pred = vmap(self.metric_fn)(p)

        codiff = vmap(self.codifferential_eta, in_axes=(0,0,0,None))(
            p, pullbacks, g_pred, params)
        codiff = jnp.squeeze(codiff)

        if full_contraction is True:
            print(f'{self.objective_function.__qualname__}: full contraction.')
            # Integrand of harmonic objective. 
            integrand = jnp.einsum('...uv, ...u, ...v -> ...', g_pred, codiff, jnp.conjugate(codiff))
            integrand = jnp.squeeze(integrand)
            return jnp.mean(jnp.abs(integrand) ** norm_order * weights)
        
        codiff_loss = jnp.mean(jnp.abs(codiff) ** norm_order * weights)
        if self.toggle_transition_loss is True:
            T = vmap(self.transition_loss, in_axes=(0,None))(p, params)
            return codiff_loss + C * jnp.mean(T)

        return codiff_loss

    def wp_metric_harmonic(self, data, eta):
        r"""
        Takes in harmonic (0,1)-T_X valued form $\eta$, contracts with holomorphic (n,0) form $\Omega$,
        then computes Weil-Petersson metric ~ $\iota{\eta} \Omega \wedge \overline{\iota{\eta} \Omega}$.
        """
        p, weights, _ = data
        weights = jnp.squeeze(weights)
        vol_Omega = jnp.mean(weights)

        # interior product with Omega (Omega is implicit in multipling by weights at the end)
        chi = jnp.einsum('i, ...ia->...ia', self.interior_product_sgn, eta)
        chi_W_chi = jnp.squeeze(jnp.einsum('ij,...ij,...ji->...', self.b_product_sgn, chi, jnp.conjugate(chi)))

        return -jnp.mean(chi_W_chi * weights) / vol_Omega

    @staticmethod
    @jit
    def cup_product(data, eta):
        p, weights, _ = data
        weights = jnp.squeeze(weights)
        vol_Omega = jnp.mean(weights)

        eta_W_eta = jnp.einsum('...uv, ...vu -> ...', eta, jnp.conjugate(eta))
        return jnp.mean(eta_W_eta * weights) / vol_Omega

    @staticmethod
    @jit
    def inner_product_Hodge(data, eta, g_pred):
        p, weights, dVol_Omega = data
        g_inv = jnp.linalg.inv(g_pred)  # g^{\bar{\nu} \mu}
        integrand = jnp.squeeze(jnp.einsum('...ua, ...vb, ...uv, ...ab', eta, jnp.conjugate(eta), g_pred, g_inv))
        vol_Omega = jnp.mean(weights)

        det_g = jnp.squeeze(jnp.real(jnp.linalg.det(g_pred)))
        vol_g = jnp.mean(det_g * weights / dVol_Omega)

        return jnp.mean(integrand * det_g * weights / dVol_Omega) / vol_g

    @partial(jit, static_argnums=(0,))
    def loss_breakdown(self, data, params):
        r"""Checks harmonicity and related conditions.
        """
        p, weights, dVol_Omega = data
        p_c = math_utils.to_complex(p)
        pullbacks = vmap(self.pb_fn)(math_utils.to_complex(p))
        g_pred = vmap(self.metric_fn)(p)

        loss = self.objective_function(data, params)
        codiff_norm = self.objective_function(data, params)

        codiff_eta = vmap(self.codifferential_eta, in_axes=(0,0,0,None))(
                p, pullbacks, g_pred, params)
        codiff_eta_mean = jnp.mean(jnp.abs(codiff_eta) * weights)

        # WP metric using Calabi-Yau isomorphism on harmonic forms
        eta, form_ref, form_correction = vmap(self.harmonic_rep_breakdown, in_axes=(0,None))(p, params)
        G_wp_harmonic = self.wp_metric_harmonic(data, eta)

        if (self.n_hyper == 1) and (len(self.ambient) == 1):
            coefficients = self.coeff_fn(self.psi)
            dQdz_info = alg_geo.dQdz_poly(self.n_homo_coords, self.monomials[0], coefficients)
            dQdz_monomials, dQdz_coeffs = dQdz_info
        else:
            dQdz_monomials, dQdz_coeffs = self.dQdz_monomials, self.dQdz_coeffs
        G_wp_KS, _ = self.compute_wp_metric_diagonal(p_c, dQdz_monomials, dQdz_coeffs, self.deformation)

        # WP metric using hodge star of bundle-valued harmonic forms
        _G_wp = self.inner_product_Hodge(data, eta, g_pred)

        polarisation_property = jnp.mean(jnp.abs(vmap(self.polarisation_preserving_property)(eta, g_pred)))
        symmetry_property = jnp.mean(jnp.abs(vmap(self.symmetry_property)(eta, g_pred)))
        transition_loss = jnp.mean(vmap(self.transition_loss, in_axes=(0,None))(p, params))

        return {"loss": loss,
                "codiff_norm": codiff_norm,
                "codiff_mean": codiff_eta_mean, 
                "transition_loss": transition_loss,
                "G_WP_CY": G_wp_harmonic,
                "G_WP_KS": G_wp_KS,
                "G_WP_bundle": _G_wp,
                "cup_product": self.cup_product(data, eta).mean(),
                "ratio (cy/bundle)": G_wp_harmonic/_G_wp,
                "<∂-bar θ, ∂-bar θ>": self.inner_product_Hodge(data, form_correction, g_pred),
                "<ξ,ξ>": self.inner_product_Hodge(data, form_ref, g_pred),
                "polarisation": polarisation_property,
                "symmetry": symmetry_property,
                "σ_measure": measures.sigma_measure(data, self.metric_fn)}


    def _idx_to_mask(self, idx):
        mask = jnp.zeros(self.n_homo_coords, dtype=bool)
        return jnp.logical_not(mask.at[idx].set(True))
    
    def compute_transition_masks(self, p):
        p_c = math_utils.to_complex(p)
        ones_mask = jnp.logical_not(jnp.isclose(p_c, jax.lax.complex(1.,0.)))
        dQdz_homo = alg_geo.evaluate_dQdz(p_c, self.dQdz_monomials, self.dQdz_coeffs)
        elim_idx, good_coord_mask_full = alg_geo.argmax_dQdz_cicy(p_c, dQdz_homo, self.n_hyper, self.n_homo_coords, True)
        good_coord_mask = good_coord_mask_full[jnp.nonzero(ones_mask, size=self.n_inhomo_coords)]
        return ones_mask, good_coord_mask_full, elim_idx
    
    def check_bounds(self, carry, _slice):
        i, state = carry
        state = jnp.logical_and(state, jnp.logical_and(self.bounds[i] <= _slice, _slice < self.bounds[i+1]))
        return (i+1, state), state

    def get_different_patches(self, elim_idx, patch_idx):
        dQ_elim_mask = self._idx_to_mask(elim_idx)
        Pi_dQ_elim_count = jnp.unique(self.proj_idx[elim_idx], return_counts=True, size=self.n_projective)[-1]
        splits = self.degrees - Pi_dQ_elim_count

        # index valid coords
        valid_coord_idx = jnp.where(dQ_elim_mask, size=self.n_homo_coords - self.n_hyper)[-1]

        factors = (valid_coord_idx,) * self.n_projective
        all_possible_patches = jnp.stack(jnp.meshgrid(*factors, indexing='ij'), axis=-1)
        all_possible_patches = jnp.reshape(all_possible_patches, (-1, self.n_projective))
        
        if self.n_projective > 1:
            # mask out invalid patches
            carry, stack = jax.lax.scan(self.check_bounds, (0, jnp.ones((self.n_homo_coords - self.n_hyper)**2, dtype=bool)),
                                                    all_possible_patches.T)
            all_possible_patches = all_possible_patches[jnp.nonzero(carry[-1], size=self.n_transitions)]

        # possibly redundant
        n_patches = all_possible_patches.shape[0]
        if n_patches != self.n_transitions:
            # pad with current patch index
            pad = jnp.tile(patch_idx, self.n_transitions - n_patches)
            pad = jnp.reshape(pad, (-1,self.n_projective,)).astype(np.int32)
            return jnp.concatenate((all_possible_patches, pad), axis=0)
            
        return all_possible_patches

    def transition_map(self, p, patch_mask, dQ_elim_mask):
        p = math_utils.to_complex(p)
        combined_mask = jnp.logical_not(patch_mask) * dQ_elim_mask

        norm = p[jnp.nonzero(patch_mask, size=self.n_projective)].reshape(-1, self.n_projective)
        if self.n_projective == 1: 
            p_ambient_transformed = p / jnp.squeeze(norm)
            return p_ambient_transformed

        all_pi_norm = 1.  # need to rescale each projective factor
        for i in range(self.n_projective):
            degrees = jnp.ones(self.degrees[i], dtype=np.complex64)
            pi_norm = degrees * norm[:,i]
            if i == 0: 
                all_pi_norm = pi_norm
            else:
                all_pi_norm = jnp.concatenate((all_pi_norm, pi_norm), axis=-1)
                
        all_pi_norm = jnp.squeeze(all_pi_norm)
        p_ambient_transformed = p / all_pi_norm
        
        return p_ambient_transformed

    @partial(jit, static_argnums=(0,))
    def Jacobian_transition_map(self, p_repeated, other_patch_mask, dQ_elim_mask):
        """
        Returns Jacobian pushforward matrix [dy^a/dx^b]_{a,b}.
        """
        # `vmap` across different patches for the same example point
        _pb = vmap(self.pb_fn)(math_utils.to_complex(p_repeated))
        combined_mask = jnp.logical_not(other_patch_mask) * dQ_elim_mask
        
        T_jac_ambient = vmap(curvature.del_z, in_axes=(0,None,0,0))(p_repeated, self.transition_map, 
                                                                other_patch_mask, dQ_elim_mask)

        T_jac = vmap(self.project_coords, in_axes=(0,0,None))(T_jac_ambient, combined_mask, self.cy_dim)
        T_jac = jnp.einsum('...ij,...aj->...ia', T_jac, _pb)
        return T_jac

    def transition_loss_vector(self, T_jac, theta_original, theta_transformed):
        theta_pushforward = jnp.einsum('...uv, ...v->...u', T_jac, theta_original)
        return theta_transformed - theta_pushforward
    
    def transition_loss(self, p, params, norm=1., max_jac_norm=10.):
        p_c = math_utils.to_complex(p)
        ones_mask, good_coord_mask_full, elim_idx = self.compute_transition_masks(p)
        patch_idx = jnp.where(jnp.logical_not(ones_mask), size=self.n_projective)[-1]
        patch_idx = jnp.reshape(patch_idx, (self.n_projective,))
        current_patch_mask = jnp.logical_not(self._idx_to_mask(patch_idx))

        if (self.n_hyper == 1) and (self.n_ambient == 1):
            other_patches = utils._generate_all_patches(self.n_homo_coords, self._transitions, self.degrees)
            other_patches = other_patches[elim_idx]  # patches with elim_idx removed.
        else:  # need to generalize to product of projective spaces
            other_patches = self.get_different_patches(elim_idx, patch_idx)

        other_patch_mask = jnp.logical_not(vmap(self._idx_to_mask)(other_patches))
        dQ_elim_mask = vmap(self._idx_to_mask)(jnp.repeat(jnp.expand_dims(elim_idx,0), self.n_transitions, axis=0))
        combined_mask = jnp.logical_not(other_patch_mask) * dQ_elim_mask

        # change the patch_idx for each point but the elim_idx is automatically fixed
        p_repeated = jnp.repeat(jnp.expand_dims(p,0), self.n_transitions, axis=0)
        # `vmap` across different patches for the same example point
        p_transformed = math_utils.to_real(vmap(self.transition_map)(p_repeated, other_patch_mask, dQ_elim_mask))

        theta_original = jnp.expand_dims(self.section_network(p, params), axis=0)
        theta_original = jnp.repeat(theta_original, self.n_transitions, axis=0)

        # [..., n_transitions, cy_dim]
        theta_transformed = vmap(self.section_network_transformed, in_axes=(0,0,None))(p_transformed, ~dQ_elim_mask, params)
        # theta_transformed = vmap(self.section_network, (0,None))(p_transformed, params)

        # J = [dy^a/dx^b]_{a,b}
        T_jac = self.Jacobian_transition_map(p_repeated, other_patch_mask, dQ_elim_mask)  # [..., n_transitions, cy_dim, cy_dim]
        jac_norm = vmap(jnp.linalg.norm)(T_jac)
        T_jac = jnp.where(jnp.expand_dims(jac_norm < max_jac_norm, (1,2)), T_jac, 0.)
        theta_original = jnp.where(jnp.expand_dims(jac_norm, 1) < max_jac_norm, theta_original, 0.)
        theta_transformed = jnp.where(jnp.expand_dims(jac_norm, 1) < max_jac_norm, theta_transformed, 0.)

        patch_diff = vmap(self.transition_loss_vector)(T_jac, theta_original, theta_transformed)
        t_loss_multipatch = jnp.abs(patch_diff) ** norm
        t_loss_multipatch = jnp.sum(t_loss_multipatch, axis=-1) / self.cy_dim
        t_loss = jnp.sum(t_loss_multipatch, axis=-1) / self.n_transitions
        
        return t_loss

    @partial(jit, static_argnums=(0,))
    def _compute_wp_metric_aux_batch_i(self, data, params):
        # internal use only
        p, weights, _ = data
        eta, form_ref, form_correction = vmap(self.harmonic_rep_breakdown, in_axes=(0,None))(p, params)
        vol_Omega_i = jnp.mean(weights)
        S_vol_Omega_i = jnp.mean((weights - vol_Omega_i)**2)

        # interior product with Omega (Omega is implicit in multipling by weights at the end)
        chi = jnp.einsum('i, ...ia->...ia', self.interior_product_sgn, eta)
        chi_W_chi = jnp.squeeze(jnp.einsum('ij,...ij,...ji->...', self.b_product_sgn, chi, jnp.conjugate(chi)))
        cWc_summand = chi_W_chi * weights
        int_cWc_i = jnp.mean(cWc_summand)
        S_int_cWc_i = jnp.mean((cWc_summand - int_cWc_i)**2)

        chi_ref = jnp.einsum('i, ...ia->...ia', self.interior_product_sgn, form_ref)
        chi_ref_W_chi_ref = jnp.squeeze(jnp.einsum('ij,...ij,...ji->...', self.b_product_sgn, chi_ref, jnp.conjugate(chi_ref)))
        chi_ref_summand = chi_ref_W_chi_ref * weights
        int_crWcr_i = jnp.mean(chi_ref_summand)
        S_int_crWcr_i = jnp.mean((chi_ref_summand - int_crWcr_i)**2)
        
        # Hodge star integrand
        g_pred = vmap(self.metric_fn)(p)
        g_inv = jnp.linalg.inv(g_pred)
        hs_integrand = jnp.squeeze(jnp.einsum('...ua, ...vb, ...uv, ...ab', eta, jnp.conjugate(eta), g_pred, g_inv))
        hs_mc_summand = hs_integrand * weights
        int_eta_HS_eta_i = jnp.mean(hs_mc_summand)
        S_int_HS_i = jnp.mean((hs_mc_summand - int_eta_HS_eta_i)**2)

        return (int_cWc_i, int_crWcr_i, int_eta_HS_eta_i, vol_Omega_i), (S_int_cWc_i, S_int_crWcr_i, S_int_HS_i, S_vol_Omega_i)
    

# TODO Register as pytree.
class HarmonicFull(Harmonic):

    def __init__(self, cy_dim: int, monomials: List[np.array], ambient: ArrayLike, 
                 deformations: List[Callable], dQdz_monomials: List[np.array], 
                 dQdz_coeffs: List[np.array], metric_fn: Callable,
                 pb_fn: Callable, coeff_fn: Callable, psi: float):
        r"""Approximation of harmonic one-forms for Calabi-Yaus with an arbitrary number of complex 
        structure moduli. Note the methods vectorise over all possible complex structure deformations.

        Parameters
        ----------
        cy_dim : int
            Dimension of Calabi-Yau manifold.
        monomials : List[np.array]
            List of defining monomials.
        ambient : array_like
            Dimensions of the ambient space factors.
        deformations : List[Callable]
            List of functions representing complex structure deformations.
        dQdz_monomials : List[np.array]
            List of monomials corresponding to polynomial Jacobian $dQ/dz$.
        dQdz_coeffs : List[np.array]
            List of coefficients corresponding to polynomial Jacobian $dQ/dz$.
        metric_fn : Callable
            Function representing metric tensor in local coordinates $g : \mathbb{R}^m -> \mathbb{C}^{a,b...}$.
        pb_fn : Callable
            Function computing pullback matrices from ambient space to projective variety.
        coeff_fn : Callable
            Function returning polynomial coefficients at given point in moduli space.
        psi : float
            Complex structure parameter psi.

        See also
        --------
        cymyc.moduli.wp.WP
        """
        super().__init__(cy_dim, monomials, ambient, deformations, dQdz_monomials, dQdz_coeffs,
                         metric_fn, pb_fn, coeff_fn, psi)
        self.h_21 = len(deformations)
        self.deformations = deformations
        self.deformation_indices = jnp.arange(self.h_21)

    def deformation_vector(self, idx, p):
        return jax.lax.switch(idx, self.deformations, p)

    def zeta_complete(self, x):
        p = math_utils.to_complex(x)
        g_FS_inv = jnp.conjugate(self._fs_metric_inverse_total(p)) # g^{\mu \bar{\nu}}
        dQdz = self._compute_dQdz_inhomo(p, self.dQdz_monomials, self.dQdz_coeffs)
        H = jnp.einsum('...ij, ...ia, ...jb->...ab', g_FS_inv, dQdz, jnp.conjugate(dQdz))

        cm_defs = vmap(self.deformation_vector, in_axes=(0,None))(self.deformation_indices, p)  # all h^{(2,1)} deformations
        dphi_dt = -jnp.einsum('...ab,...uv,...va,...cb->...cu', jnp.linalg.inv(H), g_FS_inv,
                              jnp.conjugate(dQdz), cm_defs)

        return dphi_dt  # [..., h_{21}, n_inhomo_coords]

    @partial(jit, static_argnums=(0,))
    def del_bar_zeta_complete(self, p):

        print(f'Compiling {self.del_bar_zeta_complete.__qualname__}')
        x = math_utils.to_real(p)

        dim = p.shape[-1] // 2  # complex dimension
        real_jac_x = jacfwd(self.zeta_complete)(x)
        dzeta_dx = real_jac_x[..., :dim]
        dzeta_dy = real_jac_x[..., dim:]
        dzeta_dzbar = 0.5 * (dzeta_dx + 1.j * dzeta_dy)

        return dzeta_dzbar

    def harmonic_rep_ambient(self, p, params):
        xi_ambient = self.del_bar_zeta_complete(p)  # [..., h_{21}, n_inhomo_coords, n_homo_coords]
        correction_fibre = curvature.del_bar_z(p, self.section_network, params)  # [..., h_{21}, cy_dim, n_homo_coords]
        if self.h_21 == 1: 
            correction_fibre = jnp.expand_dims(correction_fibre, axis=0)
        return xi_ambient, correction_fibre

    @partial(jit, static_argnums=(0,))
    def harmonic_rep_breakdown(self, p, params):

        p_c = math_utils.to_complex(p)
        pb = self.pb_fn(p_c)
        pb_conj = jnp.conjugate(pb)
        ones_mask, good_coord_mask = self.compute_masks(p)

        xi_ambient, correction_fibre = self.harmonic_rep_ambient(p, params)

        form_ref = vmap(self.project_coords, in_axes=(0,None,None))(xi_ambient, 
                                good_coord_mask, self.cy_dim)
        form_ref = jnp.einsum('...hai,...bi->...hab', form_ref, pb_conj)
        print(f'Compiling {self.harmonic_rep_breakdown.__qualname__}', correction_fibre.shape, xi_ambient.shape, form_ref.shape)
        form_correction = jnp.einsum('...hai,...bi->...hab', correction_fibre, pb_conj)
        eta = form_ref + form_correction

        return eta, form_ref, form_correction

    @partial(jit, static_argnums=(0,))
    def codifferential_eta(self, p:  Float[Array, "i"], pullbacks: Complex[Array, "cy_dim i"],
                           g_pred: Complex[Array, "dim dim"], params: Mapping[str, Array]) -> Complex[Array, "h_21 cy_dim"]:
        r"""Computes codifferential of $\alpha \in H^{(0,1)}(X; T_X)$ with respect
        to the given metric. This is a smooth section of the holomorphic tangent bundle,
        $\bar{\partial}^{\dagger} \eta \in \Gamma(T_X)$.
        Parameters
        ----------
        p : Float[Array, "i"]  
            2 * `complex_dim` real coords on $X$.
        pullbacks : Complex[Array, "cy_dim i"], optional
            Pullback matrices from ambient to projective variety.
        g_pred :  Complex[Array, "dim dim"]
            Predicted metric $g_{\mu \overline{\nu}}$ in local coordinates.
        params : Mapping[str, Array]
            Model parameters stored as a dictionary - keys are the module names
            registered upon initialisation and values are the parameter values.

        Returns
        -------
        codiff : Complex[Array, "h_21 cy_dim"]
            Section of tangent bundle.
        """
        _, good_coord_mask = self.compute_masks(p)
        eta = self.harmonic_rep(p, params)  # [h, a, \bar{\nu}]
        g_inv = jnp.linalg.inv(g_pred)  # g^{\bar{\nu} \mu}
        Gamma_holo = curvature.christoffel_symbols_kahler(p, self.metric_fn, pullbacks)  # [a, \kappa, b]

        # two derivatives of NN
        del_z_eta = self.del_z_harmonic_rep(p, pullbacks, good_coord_mask, params) # [h, a, \bar{\nu}, \kappa]
        # del_z_eta = self.harmonic_rep_derivative(p, params) # [a, \bar{\nu}, \kappa]
        _cov2 = jnp.einsum('...akb, ...hbv -> ...havk', Gamma_holo, eta)   # [a, \bar{\nu}, \kappa]
        covariant_derivative_eta = del_z_eta + _cov2
        codiff = jnp.einsum('...vu, ...havu->...ha', g_inv, covariant_derivative_eta)

        # don't squeeze
        return codiff # section of tangent bundle, shape [..., h_{21}, cy_dim]
    

    def objective_function(self, data, params, norm_order=1., C=10**0,
                           max_codiff_norm=10**1, full_contraction=False, norm_control=False):
        r"""Enforces condition $\bar{\partial}^{\dagger} \eta (\in C^{\infty}(X)) = 0$.
        """
        print(f'Compiling {self.objective_function.__qualname__}')

        p, weights, dVol_Omega = data
        pullbacks = vmap(self.pb_fn)(math_utils.to_complex(p))
        g_pred = vmap(self.metric_fn)(p)

        codiff = vmap(self.codifferential_eta, in_axes=(0,0,0,None))(
            p, pullbacks, g_pred, params)
        print(f'{self.objective_function.__qualname__}, codiff shape {codiff.shape}')

        if norm_control is True:
            print(f'{self.objective_function.__qualname__}, controlling norm, shape {codiff_norm.shape}')
            codiff_norm = vmap(jnp.linalg.norm)(codiff) / self.h_21  # don't squeeze
            codiff = jnp.where(jnp.expand_dims(codiff_norm, (1,2)) < max_codiff_norm, codiff, 0.)

        if full_contraction is True:
            print(f'{self.objective_function.__qualname__}: full contraction.')
            # Integrand of harmonic objective. 
            integrand = jnp.einsum('...uv, ...hu, ...hv -> ...', g_pred, codiff, jnp.conjugate(codiff))
            integrand = jnp.squeeze(integrand)
            return jnp.mean(jnp.abs(integrand) ** norm_order * weights)

        abs_codiff_contract = jnp.mean(jnp.abs(codiff), axis=(1,2)) ** norm_order
        
        if self.toggle_transition_loss is True:
            T = vmap(self.transition_loss, in_axes=(0,None))(p, params)
            return jnp.mean(abs_codiff_contract * weights) + C * jnp.mean(T)

        return jnp.mean(abs_codiff_contract * weights)

    @partial(jit, static_argnums=(0,))
    def loss_breakdown(self, data, params):
        r"""Checks harmonicity and related conditions.
        """
        p, weights, dVol_Omega = data
        p_c = math_utils.to_complex(p)
        pullbacks = vmap(self.pb_fn)(math_utils.to_complex(p))
        g_pred = vmap(self.metric_fn)(p)

        loss = self.objective_function(data, params)
        codiff_eta = vmap(self.codifferential_eta, in_axes=(0,0,0,None))(
                p, pullbacks, g_pred, params)
        codiff_eta_mean = jnp.mean(jnp.abs(codiff_eta))

        # WP metric using Calabi-Yau isomorphism on harmonic forms
        eta, form_ref, form_correction = vmap(self.harmonic_rep_breakdown, in_axes=(0,None))(p, params)
        G_wp_harmonic = self.wp_metric_harmonic(data, eta)
        cup_product = self.cup_product(data, eta)
        G_wp_KS = self.compute_wp_metric_complete(p_c)  # special geometry computation
        # WP metric using hodge star of bundle-valued harmonic forms
        _G_wp = self.inner_product_Hodge(data, eta, g_pred)

        print(f'{self.loss_breakdown.__qualname__}: {eta.shape}, {form_ref.shape}, {form_correction.shape}')
        print(f'{self.loss_breakdown.__qualname__}: {G_wp_harmonic.shape}, {cup_product.shape}')
        print(f'{self.loss_breakdown.__qualname__}: {G_wp_KS.shape}, {_G_wp.shape}')

        polarisation_property = jnp.mean(jnp.abs(vmap(
            vmap(self.polarisation_preserving_property, in_axes=(0,None)))(eta, g_pred)))
        symmetry_property = jnp.mean(jnp.abs(vmap(
            vmap(self.symmetry_property, in_axes=(0,None)))(eta, g_pred)))
        transition_loss = jnp.mean(vmap(self.transition_loss, in_axes=(0,None))(p, params))

        G_wp_harmonic_diag = jnp.diag(G_wp_harmonic)
        cup_product_diag = jnp.diag(cup_product)
        G_wp_KS_diag = jnp.diag(G_wp_KS)
        hs_diag = jnp.diag(_G_wp)

        return {"loss": loss,
                "codiff_mean": codiff_eta_mean, 
                "transition_loss": transition_loss,

                "G_WP_CY": G_wp_harmonic_diag,
                "cup_product": cup_product_diag,
                "G_WP_KS": G_wp_KS_diag,
                "G_WP_bundle": hs_diag,
                "ratio (cy/bundle)": G_wp_harmonic_diag/hs_diag,

                "(ξ,ξ) (WP)": jnp.diag(self.cup_product(data, form_ref)),
                "(∂-bar θ, ∂-bar θ) (WP)": jnp.diag(self.cup_product(data, form_correction)),

                "polarisation": polarisation_property,
                "symmetry": symmetry_property,
                "σ_measure": measures.sigma_measure(data, self.metric_fn)}

    @partial(jit, static_argnums=(0,))
    def zeta_jacobian_complete(self, p):

        print(f'Compiling {self.zeta_jacobian_complete.__qualname__}')
        x = math_utils.to_real(p)

        dim = p.shape[-1]  # complex dimension
        real_jac_x = jacfwd(self.zeta_complete)(x)
        dzeta_dx = real_jac_x[..., :dim]
        dzeta_dy = real_jac_x[..., dim:]
        dzeta_dz = 0.5 * (dzeta_dx - 1.j * dzeta_dy)
        dzeta_dzbar = 0.5 * (dzeta_dx + 1.j * dzeta_dy)

        return dzeta_dz, dzeta_dzbar

    @partial(jit, static_argnums=(0,))
    def compute_wp_metric_complete(self, p): 
        """
        Computes WP metric obtained by MC integration over fibre points `p`.
        See Mirror Symmetry, Mori, eq. (6.1).
        NB: Don't `vmap` this.
        """

        weights, pb, *_ = vmap(self.integration_weights_fn, in_axes=(0,None,None))(
            p, self.dQdz_monomials, self.dQdz_coeffs)
        vol_Omega = jnp.mean(weights)
        ones_mask = jnp.logical_not(jnp.isclose(p, jax.lax.complex(1.,0.)))

        # get Lie derivative of Ω along all diffeomorphism directions [..., h_{21}, n_inhomo_coords, n_homo_coords]
        dzeta_dz, dzeta_dzbar = vmap(self.zeta_jacobian_complete)(p)

        B = vmap(vmap(self.compute_bij, in_axes=(None,0,None,None,None,None)),
                 in_axes=(0,0,0,0,None,None))(p, dzeta_dzbar, pb, ones_mask, self.dQdz_monomials, self.dQdz_coeffs)
        B = jnp.squeeze(B, axis=2)
        B_w_B = jnp.einsum('ij,...aij,...bji->...ab', self.b_product_sgn, B, jnp.conjugate(B))

        _A = vmap(vmap(self.project_to_good_ambient_coords, in_axes=(0,None)), in_axes=(0,0))(
            jnp.einsum('...ij->...ji',dzeta_dz), ones_mask)
        _A = jnp.squeeze(_A, axis=2)
        A = -jnp.einsum('...ii->...', _A)
        A_w_A = jnp.einsum('...a,...b->...ab', A, jnp.conjugate(A))

        int_A = jnp.mean(jnp.expand_dims(weights, axis=1) * A, axis=0)
        G_wp = -jnp.mean(jnp.expand_dims(weights, axis=(1,2)) * (A_w_A + B_w_B), axis=0) / vol_Omega + \
            jnp.einsum('...a, ...b->...ab', jnp.conjugate(int_A), int_A) / vol_Omega**2

        return G_wp

    @partial(jit, static_argnums=(0,))
    def kappa_complete(self, p, dQdz_monomials, dQdz_coeffs, weights=None, pb=None, output_variance=True):
        r"""
        Yukawa couplings for a CY threefold. \int_{X_s} \Omega \wedge \frac{d^3 \Omega}{ds^a ds^b ds^c} \vert_{s=0}.
        """

        if weights is None:
            weights, pb, *_ = vmap(self.integration_weights_fn, in_axes=(0,None,None))(
                p, dQdz_monomials, dQdz_coeffs)

        ones_mask = jnp.logical_not(jnp.isclose(p, jax.lax.complex(1.,0.)))
        dQdz_homo = vmap(alg_geo.evaluate_dQdz, in_axes=(0,None,None))(p, dQdz_monomials, dQdz_coeffs)
        Omega = vmap(alg_geo._holomorphic_volume_form, in_axes=(0,0,None,None,None))(
            p, dQdz_homo, self.n_hyper, self.n_homo_coords, self.ambient)

        # get Lie derivative of Ω along all diffeomorphism directions [..., h_{21}, n_inhomo_coords, n_homo_coords]
        _, dzeta_dzbar = vmap(self.zeta_jacobian_complete)(p)
        # pullback to CY
        dzeta_dzbar_pb = vmap(vmap(self.pullback_diffeo_jacobian, in_axes=(None,0,None,None,None)))(
            p, dzeta_dzbar, pb, ones_mask, dQdz_homo)
        dzeta_dzbar_pb = jnp.squeeze(dzeta_dzbar_pb)

        contraction = jnp.einsum('...ijk, ...xyz, ...aix, ...bjy, ...ckz -> ...abc',
                   self.eps_3d, self.eps_3d, dzeta_dzbar_pb, dzeta_dzbar_pb,
                   dzeta_dzbar_pb)
        contraction = jnp.squeeze(contraction)

        kappa_abc = jnp.expand_dims(Omega**2, axis=((1,2,3))) * contraction
        dVol_Omega = jnp.real(Omega * jnp.conjugate(Omega))

        kappa_integrand = jnp.expand_dims(weights / dVol_Omega, axis=((1,2,3))) * kappa_abc
        int_kappa_abc = jnp.mean(kappa_integrand, axis=0)

        if output_variance is True:
            S_int_re_kappa = jnp.mean((jnp.real(kappa_integrand) - jnp.real(int_kappa_abc))**2, axis=0)
            S_int_im_kappa = jnp.mean((jnp.imag(kappa_integrand) - jnp.imag(int_kappa_abc))**2, axis=0)
            return int_kappa_abc, S_int_re_kappa, S_int_im_kappa

        return int_kappa_abc
    
    @partial(jit, static_argnums=(0,))
    def _compute_wp_complete_batch(self, p, weights, pb): 
        """
        Computes (r,s) element of WP metric obtained by MC integration over fibre points `p`. 
        Tangent vectors defined by r,s indices arguments. See Mirror Symmetry, Mori, eq. (6.1).
        Yields mean over batch, to be processed using incremental averaging.
        NB: Don't `vmap` this.
        """
        ones_mask = jnp.logical_not(jnp.isclose(p, jax.lax.complex(1.,0.)))

        # get Lie derivative of Ω along all diffeomorphism directions [..., h_{21}, n_inhomo_coords, n_homo_coords]
        dzeta_dz, dzeta_dzbar = vmap(self.zeta_jacobian_complete)(p)

        B = vmap(vmap(self.compute_bij, in_axes=(None,0,None,None,None,None)),
                 in_axes=(0,0,0,0,None,None))(p, dzeta_dzbar, pb, ones_mask, self.dQdz_monomials, self.dQdz_coeffs)
        B = jnp.squeeze(B)
        B_w_B = jnp.einsum('ij,...aij,...bji->...ab', self.b_product_sgn, B, jnp.conjugate(B))

        _A = vmap(vmap(self.project_to_good_ambient_coords, in_axes=(0,None)), in_axes=(0,0))(
            jnp.einsum('...ij->...ji',dzeta_dz), ones_mask)
        _A = jnp.squeeze(_A)
        A = -jnp.einsum('...ii->...', _A)
        A_w_A = jnp.einsum('...a,...b->...ab', A, jnp.conjugate(A))

        weights, A, A_w_A, B_w_B = jnp.squeeze(weights), jnp.squeeze(A), jnp.squeeze(A_w_A), jnp.squeeze(B_w_B)

        vol_Omega_i = jnp.mean(weights)
        S_vol_Omega_i =  jnp.mean((weights - vol_Omega_i)**2)

        wA = jnp.expand_dims(weights, axis=1) * A
        int_A_i = jnp.mean(wA, axis=0)
        S_int_A_i = jnp.mean(jnp.square(wA - int_A_i), axis=0)

        wApB = jnp.expand_dims(weights, axis=(1,2)) * (A_w_A + B_w_B)
        int_AB_i = jnp.mean(wApB, axis=0)
        S_int_AB_i = jnp.mean(jnp.square(wApB - int_AB_i), axis=0)

        return (vol_Omega_i, int_A_i, int_AB_i), (S_vol_Omega_i, S_int_A_i, S_int_AB_i)
    
    def transition_loss(self, p, params, norm=1., max_jac_norm=10**1):
        p_c = math_utils.to_complex(p)
        ones_mask, good_coord_mask_full, elim_idx = self.compute_transition_masks(p)
        patch_idx = jnp.where(jnp.logical_not(ones_mask), size=self.n_projective)[-1]
        patch_idx = jnp.reshape(patch_idx, (self.n_projective,))
        current_patch_mask = jnp.logical_not(self._idx_to_mask(patch_idx))

        if (self.n_hyper == 1) and (self.n_ambient == 1):
            other_patches = utils._generate_all_patches(self.n_homo_coords, self._transitions, self.degrees)
            other_patches = other_patches[elim_idx]  # patches with elim_idx removed.
        else:  # need to generalize to product of projective spaces
            other_patches = self.get_different_patches(elim_idx, patch_idx)

        other_patch_mask = jnp.logical_not(vmap(self._idx_to_mask)(other_patches))
        dQ_elim_mask = vmap(self._idx_to_mask)(jnp.repeat(jnp.expand_dims(elim_idx,0), self.n_transitions, axis=0))
        combined_mask = jnp.logical_not(other_patch_mask) * dQ_elim_mask

        # change the patch_idx for each point but the elim_idx is automatically fixed
        p_repeated = jnp.repeat(jnp.expand_dims(p,0), self.n_transitions, axis=0)
        # `vmap` across different patches for the same example point
        p_transformed = math_utils.to_real(vmap(self.transition_map)(p_repeated, other_patch_mask, dQ_elim_mask))

        theta_original = jnp.expand_dims(self.section_network(p, params), axis=0)
        theta_original = jnp.repeat(theta_original, self.n_transitions, axis=0)

        # [..., n_transitions, h_21, cy_dim]
        theta_transformed = vmap(self.section_network_transformed, in_axes=(0,0,None))(p_transformed, ~dQ_elim_mask, params)
        # theta_transformed = vmap(self.section_network, (0,None))(p_transformed, params)

        # J = [dy^a/dx^b]_{a,b}
        T_jac = self.Jacobian_transition_map(p_repeated, other_patch_mask, dQ_elim_mask)  # [..., n_transitions, cy_dim, cy_dim]
        jac_norm = vmap(jnp.linalg.norm)(T_jac)
        T_jac = jnp.where(jnp.expand_dims(jac_norm < max_jac_norm, (1,2)), T_jac, 0.)
        theta_original = jnp.where(jnp.expand_dims(jac_norm, axis=(1,2)) < max_jac_norm, theta_original, 0.)
        theta_transformed = jnp.where(jnp.expand_dims(jac_norm, axis=(1,2)) < max_jac_norm, theta_transformed, 0.)

        # [..., n_transitions, h_{21},  cy_dim]
        patch_diff = vmap(vmap(self.transition_loss_vector, in_axes=(None,0,0)))(T_jac, theta_original, theta_transformed)
        t_loss_multipatch = jnp.abs(patch_diff) ** norm

        t_loss_multipatch = jnp.sum(t_loss_multipatch, axis=-1) / self.cy_dim
        t_loss = jnp.sum(t_loss_multipatch, axis=-1) / self.h_21
        t_loss = jnp.sum(t_loss, axis=-1) / self.n_transitions
        
        print(f'{self.transition_loss.__qualname__}: {theta_transformed.shape}, {theta_original.shape}')
        print(f'{self.transition_loss.__qualname__}: {T_jac.shape}')
        print(f'{self.transition_loss.__qualname__}: {t_loss_multipatch.shape}')
        return t_loss
    

    def __call__(self, p: Float[Array, "i"], params: Mapping[str, Array]) -> Complex[Array, "h_21 cy_dim cy_dim"]:
        r"""
        Constructs all harmonic representatives by $\overline{\partial}$-exact correction to a
        representative from the $H^{0,1}$ Dolbeault cohomology, $\xi$;

        $$ 
        \eta = \xi + \overline{\partial} \theta~.
        $$

        Here $\theta$ is taken to be a linear combination of a basis of sections of $V$.
        Parameters
        ----------
        p : Float[Array, "i"]  
            2 * `complex_dim` real coords on $X$.
        params : Mapping[str, Array]
            Model parameters stored as a dictionary - keys are the module names
            registered upon initialisation and values are the parameter values.
        """        
        return self.harmonic_rep(p, params)

    @staticmethod
    @jit
    def inner_product_Hodge(data: Tuple[ArrayLike, ArrayLike, ArrayLike], eta: Complex[Array, "h_21 cy_dim cy_dim"],
                            g_pred: Complex[Array, "cy_dim cy_dim"]) -> Complex[Array, "h_21 h_21"]:
        r"""
        Hodge star inner product between harmonic forms `eta` parameterising moduli tangent directions to yield Weil-Petersson metric,

        $$ 
        \mathcal{G}_{a\overline{b}} \propto \int_X \eta_a \wedge \overline{\star}_g \eta_b~.
        $$
        
        This should agree with the cup product calculation provided the Ricci-flat metric is used.
        Parameters
        ----------
        data : Tuple[ArrayLike, ArrayLike, ArrayLike]
            Tuple containing input points, integration weights and canonical volume form
            $\Omega \wedge \bar{\Omega}$ in local coords.
        eta  : Complex[Array, "h cy_dim cy_dim"])
            Harmonic representative $\eta$.
        g_pred : Complex[Array, "cy_dim cy_dim"]
            Approximate Ricci-flat metric in local coords.

        Returns
        -------
        Complex[Array, "h_21 h_21"]
            Weil-Petersson metric.
        """
        p, weights, dVol_Omega = data
        g_inv = jnp.linalg.inv(g_pred)  # g^{\bar{\nu} \mu}
        # No squeeze
        integrand = jnp.einsum('...iua, ...jvb, ...uv, ...ab->...ij', eta, jnp.conjugate(eta), g_pred, g_inv)
        vol_Omega = jnp.mean(weights)

        det_g = jnp.squeeze(jnp.real(jnp.linalg.det(g_pred)))
        vol_g = jnp.mean(det_g * weights / dVol_Omega)

        _weights = jnp.expand_dims(det_g * weights / dVol_Omega, axis=(1,2))
        return jnp.mean(integrand * _weights, axis=0) / vol_g

    def wp_metric_harmonic(self, data: Tuple[ArrayLike, ArrayLike, ArrayLike], 
                           eta: Complex[Array, "h_21 cy_dim cy_dim"]) -> Complex[Array, "h_21 h_21"]:
        r"""
        Takes in harmonic one-form `eta` and forms interior product with the holomorphic form $\Omega$,
        to yield Weil-Petersson metric via cup product,

        $$ 
        \mathcal{G}_{a\overline{b}} \propto \int_X \iota_{\eta_a} \Omega \wedge \overline{\iota_{\eta_b} \Omega}~.
        $$
        
        Parameters
        ----------
        data : Tuple[ArrayLike, ArrayLike, ArrayLike]
            Tuple containing input points, integration weights and canonical volume form
            $\Omega \wedge \bar{\Omega}$ in local coords.
        eta  : Complex[Array, "h cy_dim cy_dim"])
            Harmonic representative $\eta$.

        Returns
        -------
        Complex[Array, "h_21 h_21"]
            Weil-Petersson metric.
        """

        p, weights, _ = data
        weights = jnp.squeeze(weights)
        vol_Omega = jnp.mean(weights)

        # interior product with Omega (Omega is implicit in multipling by weights at the end)
        chi = jnp.einsum('i, ...hia->...hia', self.interior_product_sgn, eta)
        # No squeeze
        chi_W_chi = jnp.einsum('ij,...aij,...bji->...ab', self.b_product_sgn, chi, jnp.conjugate(chi))

        return -jnp.mean(chi_W_chi * jnp.expand_dims(weights, axis=(1,2)), axis=0) / vol_Omega

    @staticmethod
    @jit
    def cup_product(data, eta):
        p, weights, _ = data
        weights = jnp.squeeze(weights)
        vol_Omega = jnp.mean(weights)

        eta_W_eta = jnp.einsum('...auv, ...bvu -> ...ab', eta, jnp.conjugate(eta))
        return jnp.mean(eta_W_eta * jnp.expand_dims(weights, axis=(1,2)), axis=0) / vol_Omega

    # INTERNAL USE ONLY - only used to generate plot!
    @partial(jit, static_argnums=(0,))
    def _compute_wp_metric_harmonic_aux_batch_i(self, data, params):
        r"""Internal use only.
        Takes in harmonic $(0,1)-T_X$ valued form $\eta$, contracts with holomorphic (n,0) form $\Omega$,
        then computes Weil-Petersson metric ~ $\iota{\eta} \Omega \wedge \overline{\iota{\eta} \Omega}$.
        """

        p, weights, _ = data
        g_pred = vmap(self.metric_fn)(p)
        g_inv = jnp.linalg.inv(g_pred)
        eta, form_ref, form_correction = vmap(self.harmonic_rep_breakdown, in_axes=(0,None))(p, params)
        weights = jnp.squeeze(weights)
        vol_Omega_i = jnp.mean(weights)
        _w = jnp.expand_dims(weights, axis=(1,2))

        chi_W_chi = jnp.einsum('...auv, ...bvu -> ...ab', eta, jnp.conjugate(eta))
        cWc_integrand = chi_W_chi * _w
        int_cWc_i = jnp.mean(cWc_integrand, axis=0)
        S_int_cWc_i = jnp.mean(jnp.square(cWc_integrand - int_cWc_i), axis=0)

        ref_W_ref = jnp.einsum('...auv, ...bvu -> ...ab', form_ref, jnp.conjugate(form_ref))
        crWcr_integrand = ref_W_ref * _w
        int_crWcr_i = jnp.mean(crWcr_integrand, axis=0)
        S_int_crWcr_i = jnp.mean(jnp.square(crWcr_integrand - int_crWcr_i), axis=0)

        # squeeze this?
        hs_integrand = jnp.einsum('...iua, ...jvb, ...uv, ...ab->...ij', eta, jnp.conjugate(eta), g_pred, g_inv)
        hs_mc_summand = hs_integrand * _w
        int_eta_HS_eta_i = jnp.mean(hs_mc_summand, axis=0)
        S_int_eta_HS_eta_i = jnp.mean(jnp.square(hs_mc_summand - int_eta_HS_eta_i), axis=0)

        return (int_cWc_i, int_crWcr_i, int_eta_HS_eta_i, vol_Omega_i), (S_int_cWc_i, S_int_crWcr_i, S_int_eta_HS_eta_i)
