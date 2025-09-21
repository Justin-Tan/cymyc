import jax
jax.config.update("jax_enable_x64", True)

import numpy as np  # original CPU-backed NumPy
import jax.numpy as jnp

from jax import jit, jacfwd, vmap, random
import optax

from functools import partial

import math, time, argparse, os
import sympy as sp

import time
# from tqdm import tqdm
from tqdm.notebook import tqdm

# custom
from cymyc.utils import math_utils, poly_utils, pointgen
from cymyc.utils import gen_utils as utils
from cymyc import alg_geo, fubini_study, curvature
from cymyc.approx import models, hym
from cymyc.approx.train import create_train_state

import cymyc.dataloading as dataloading


import sympy as sp
from flax import linen as nn

import sympy as sp
from flax import linen as nn
from collections import defaultdict

class HarmonicBundle:

    def __init__(self, metric_fn, monomials, coefficients, cy_dim, ambient, defining_polys=None):
        
        self.monomials, self.coefficients = monomials, coefficients
        self.ambient = ambient
        self.ambient_dim = sum(self.ambient)
        self.cy_dim = cy_dim
        self._slope = None
        self._metric_fn = metric_fn

        # specify monad data
        # make arguments later
        # self.family_ids = [0,2,6,8,17,19,22,40,42,45,49]
        self.family_ids = [2,6,8,22,40,42,45,49]

        # self.n_harmonic = len(self.family_ids)
        self.n_harmonic = 1
        self.rank_V = 3
        self.twisting_degree = 4
        self.line_bundle_B = (1,1,1,1)
        self.rank_B = len(self.line_bundle_B)
        self.line_bundle_C = (4,)
        self.mb1 = jnp.asarray(poly_utils.monomial_basis(ambient, 1))
        self.mb3 = jnp.asarray(poly_utils.monomial_basis(ambient, 3)) # for basis of sections of $V \otimes O_X(k)$
        self.mb4 = jnp.asarray(poly_utils.monomial_basis(ambient, 4)) # for untwisting sections
        self.cdtype = np.complex64
        self.n_linear = len(self.mb1)
        # self.N_sb = len(self.mb1) * self.rank_B # 3  # number of sections of $E$

        self.n_hyper = self.ambient_dim - self.cy_dim
        self.n_homo_coords = monomials.shape[-1]
        self.degrees = ambient + 1
        self.n_inhomo_coords = sum(self.degrees) - len(self.degrees)
        dQdz_info = alg_geo.dQdz_poly(self.n_homo_coords, monomials, coefficients)
        self.dQdz_monomials, self.dQdz_coeffs = dQdz_info
        self.fs_metric_fn = jax.tree_util.Partial(fubini_study.fubini_study_metric_homo_pb, 
                                                  dQdz_info=(self.dQdz_monomials, self.dQdz_coeffs), cy_dim=cy_dim)
        self.pb_fn = partial(alg_geo.compute_pullbacks,
                    dQdz_info=(self.dQdz_monomials, self.dQdz_coeffs),
                    cy_dim=self.cy_dim, cdtype=self.cdtype)
        
        self.Omega_fn = partial(alg_geo._holomorphic_volume_form, 
                                n_hyper=self.n_hyper, n_coords=self.n_homo_coords,
                                ambient=self.ambient)

        self.log_H_ref_fn = partial(hym.reference_hermitian_structure, 
                                    line_bundle=tuple((self.line_bundle_B[0],)), 
                                    ambient=tuple(self.ambient))
        
        if defining_polys is None:  # projective space
            self.monomial_basis = poly_utils.MonomialBasis(ambient, self.twisting_degree)
        else:
            self.monomial_basis = poly_utils.MonomialBasisReduced(ambient, self.twisting_degree, defining_polys)

        self.all_mono_eval_fn = jax.tree_util.Partial(poly_utils.monomial_evaluate_log, 
                                                      s_k=self.monomial_basis.power_matrix, 
                                                      conj=False)

        self.n_Vk = self.rank_B * poly_utils.dim_OXk(self.ambient, self.twisting_degree-1, self.monomial_basis.mod_degree)
        self.n_Ok = poly_utils.dim_OXk(self.ambient, self.twisting_degree, self.monomial_basis.mod_degree)
        self.N_sb = self.n_Vk
        
        mbl, mbq = poly_utils.MonomialBasis(ambient, 1), poly_utils.MonomialBasis(ambient, 2)
        variables = sp.symarray('z', ambient.item() + len(ambient))
        monad_map = [v**3 for v in variables[:4]]
        self.monad_map_power_matrix = poly_utils.monomials_to_power_matrix(monad_map, variables)
        monomials_B = mbl.power_matrix
        monomials_C = self.monomial_basis.power_matrix
        self.quotient_basis, ideal_generators, groebner_basis = poly_utils.get_quotient_basis(variables, monad_map, 
                                                                                   monomials_B, monomials_C)
        
        self.eps_3d = jnp.array(math_utils.n_dim_eps_symbol(3))
        self.activation = nn.gelu

        _monad_map = [v for v in variables[:4]]
        self.monad_map_power_matrix_DKLR = poly_utils.monomials_to_power_matrix(_monad_map, variables)
        self._N_sb = 19

        self.conf_mat, p_conf_mat = math_utils._configuration_matrix([monomials], ambient)
        self.t_degrees = math_utils._find_degrees(self.ambient, self.n_hyper, self.conf_mat)
        self.kmoduli_ambient = math_utils._kahler_moduli_ambient_factors(self.cy_dim, self.ambient, self.t_degrees)
        
        if (self.n_hyper > 1) or (len(self.ambient) > 1):
            self.integration_weights_fn = partial(alg_geo._integration_weights_cicy, 
                dQdz_monomials=self.dQdz_monomials, dQdz_coefficients=self.dQdz_coeffs,                              
                n_hyper=self.n_hyper, cy_dim=self.dim, n_coords=self.n_homo_coords,
                ambient=self.ambient, kmoduli_ambient=self.kmoduli_ambient, cdtype=self.cdtype)
        else:
            self.integration_weights_fn = partial(alg_geo.compute_integration_weights,
                                                  dQdz_monomials=self.dQdz_monomials,
                                                  dQdz_coefficients=self.dQdz_coeffs,
                                                  cy_dim=self.cy_dim)

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
        
        ddbar_f = 0.25 * jnp.squeeze((d2f_dx2 + d2f_dy2) -  1.j * (d2f_dxdy - d2f_dydx))
        return ddbar_f

    @staticmethod
    def dagger(A):
        return jnp.einsum("...ab->...ba", jnp.conjugate(A))

    @partial(jax.jit, static_argnums=(0,2))
    def connection_form_V(self, p, H_fn, params=None):
        if params is None:
            H = H_fn(p)
            del_H = curvature.del_z(p, H_fn)
        else:
            H = H_fn(p, params)
            del_H = curvature.del_z(p, H_fn, False, params)
        H_inv = jnp.linalg.inv(H)  # \bar{a} b
        pb = self.pb_fn(math_utils.to_complex(p))
        del_H = jnp.einsum("...abu,...iu->...abi", del_H, pb)
        A = jnp.einsum("...bc, ...abi->...cai", H_inv, del_H)  # A^c_{ai}
        return A

    @partial(jax.jit, static_argnums=(0,2))
    def curvature_form_V(self, p, H_fn, params=None):
        F = curvature.del_bar_z(p, self.connection_form_V, False, H_fn, 
                                params)
        pb = self.pb_fn(math_utils.to_complex(p))
        F = jnp.einsum("...abiu, ...ju->...abij", F, jnp.conjugate(pb))
        return F

    def fubini_study_metric_B(self, p, cdtype=np.complex64):
        r"""FS reference metric on a direct sum of line bundles, e.g. on 
        bundle $B$.
        """
        log_H = self.log_H_ref_fn(p)
        H_fs = jnp.eye(self.rank_B, dtype=cdtype) * jnp.exp(log_H)
        return H_fs

    def fubini_study_metric_V(self, p, cdtype=np.complex64):
        r"""FS reference metric on subbundle $\iota V: \righthookarrow B$ of
        direct sum of line bundles.
        """
        H_fs_ambient = self.fubini_study_metric_B(p)
        _embedding = self.embedding_matrix(p)
        return jnp.einsum("...ab, ...ia, ...jb->...ij", H_fs_ambient, _embedding, 
                          jnp.conjugate(_embedding))

    @partial(jax.jit, static_argnums=(0,2))
    def fubini_study_metric_twisted_dual(self, p, k, cdtype=np.complex64):
        r"""Twisted FS reference metric $V^{\vee} \otimes O_X(k)$
        """
        H_fs_V = self.fubini_study_metric_V(p)
        H_fs_V_dual = jnp.linalg.inv(H_fs_V)  # \bar{\mu} \nu
        
        H_fs_Ok = hym.reference_hermitian_structure(p, (k,), tuple(self.ambient))
        
        return H_fs_V_dual * H_fs_Ok  # \bar{\mu} \nu


    def preimage_monomials(self, q_basis_element):
        return jnp.expand_dims(q_basis_element,0) - self.monad_map_power_matrix
    
    def partition_of_unity(self, p):
        p_c = math_utils.to_complex(p)[:len(self.monad_map_power_matrix)]
        exp_arg = jnp.real(p_c * jnp.conjugate(p_c))
        p_abs_sq = jnp.sum(exp_arg)
        w = jnp.exp(-exp_arg / p_abs_sq)
        return w / jnp.sum(w)
    
    def monad_map_preimage(self, p):
        r"""
        Preimage of monad map, giving smooth sections of $C^{\infty}(X; B)
        s.t. f(preimage) = quotient_mono(p) for each quotient mono in the basis. 
        This is $\hat{\mu}$ in the LES. 
        """
        _preimage_monomials = vmap(self.preimage_monomials)(self.quotient_basis)
        _preimage_coeffs = self.partition_of_unity(p)
        mono_eval = poly_utils.monomial_evaluate_log(p, _preimage_monomials)
        return jnp.expand_dims(_preimage_coeffs, axis=(0,)) * mono_eval
        
    def del_bar_section_B(self, p):
        del_bar_mu = curvature.del_bar_z(p, self.monad_map_preimage)
        pb = self.pb_fn(math_utils.to_complex(p))
        return jnp.einsum("...hav, ...uv->...hau", del_bar_mu, jnp.conjugate(pb))
    
    def monad_map(self, p, s_B):
        """
        Explicit monad map on smooth sections of bundle B
        Example usage:
        monad_image = vmap(vmap(monad_map, in_axes=(None,0)), in_axes=(0,0))(p, s_B)
        """
        f_p = poly_utils.monomial_evaluate_log(p, self.monad_map_power_matrix)
        return jnp.sum(f_p * s_B)
        
    def embedding_matrix(self, p):
        r"""
        Describes embedding $\iota: V \righthookarrow B$.
        """
        patch_idx = 0  # jnp.argmax(jnp.abs(math_utils.to_complex(p))[:self.rank_B])
        proj = jnp.eye(self.rank_V, dtype=self.cdtype)
        f_p = poly_utils.monomial_evaluate_log(p, self.monad_map_power_matrix)
        col = -f_p # / f_p[patch_idx]
        col = jnp.delete(col, patch_idx, assume_unique_indices=True)
        return jnp.insert(proj, patch_idx, col, axis=-1)

    def projection_matrix(self, p):
        r"""
        Inverse of embedding $\iota: V \righthookarrow B$.
        """
        patch_idx = jnp.argmax(jnp.abs(math_utils.to_complex(p))[:self.rank_B])
        proj = jnp.eye(self.rank_V, dtype=self.cdtype)
        col = jnp.zeros(self.rank_V, dtype=self.cdtype)
        return jnp.insert(proj, patch_idx, col, axis=-1)

    def section_basis(self, p):
        """
        Section basis described in ambient coordinates, expressed in a local
        frame $Z_i$ on $U_i$. 
        """
        # return self.embedding_matrix(p)
        p_c = math_utils.to_complex(p)
        patch_idx = jnp.argmax(jnp.abs(p_c)[:self.rank_B])

        proj = jnp.eye(self.rank_V, dtype=self.cdtype)
        f_p = poly_utils.monomial_evaluate_log(p, self.monad_map_power_matrix)
        col = -f_p / f_p[patch_idx]
        col = jnp.delete(col, patch_idx, assume_unique_indices=True)
        return jnp.insert(proj, patch_idx, col, axis=-1) * p_c[patch_idx]
          
    
    def twisted_section_basis(self, p, cdtype=np.complex64):
        r"""
        Holomorphic sections of twisted dual bundle $V^{\vee} \otimes O_X(k)$,
        expressed in a local frame - typically Z_i^k. 
        """
        p_c = math_utils.to_complex(p)
        patch_idx = 0  #jnp.argmax(jnp.abs(p_c)[:self.rank_B])
        
        # section_matrix = jnp.zeros((self.rank_B, len(self.mb3) * self.rank_B), dtype=cdtype)
        # Ok_powers = self.mb3.at[:,patch_idx].subtract(3)
        Ok_powers = self.mb3
        Ok_monomials = poly_utils.monomial_evaluate_log(p, Ok_powers)
    
        blocks = [Ok_monomials] * self.rank_B
        section_matrix = jax.scipy.linalg.block_diag(*blocks)
        embedding_matrix = self.embedding_matrix(p)
    
        return embedding_matrix @ section_matrix


    @partial(jax.jit, static_argnums=(0,))
    def H1XV_representatives(self, p):
        """
        Representatives of the $H^1(X;V)$ cohomology
        """
        patch_idx = jnp.argmax(jnp.abs(math_utils.to_complex(p))[:self.rank_B])
        nu = self.del_bar_section_B(p)
        # project onto subbundle
        nu = jnp.delete(nu, patch_idx, axis=-2, assume_unique_indices=True)
        # nu = jnp.delete(nu, 0, axis=-2, assume_unique_indices=True)

        # select families for testing
        return jnp.take(nu, np.asarray(self.family_ids), axis=0)


    @partial(jax.jit, static_argnums=(0,))
    def yukawa_couplings(self, p):
        p_c = math_utils.to_complex(p)
        weights, pb, dVol_Omega, _ = vmap(self.integration_weights_fn)(p_c)

        dQdz = vmap(alg_geo.evaluate_dQdz, in_axes=(0,None,None))(p_c, self.dQdz_monomials, self.dQdz_coeffs)
        Omega = vmap(self.Omega_fn)(p_c, jnp.expand_dims(dQdz,1))

        nu = vmap(self.H1XV_representatives)(p)  # [..., h^1_V, rank_V, cy_dim]
        
        contraction = jnp.einsum('...ijk, ...xyz, ...aix, ...bjy, ...ckz -> ...abc',
                   self.eps_3d, self.eps_3d, nu, nu, nu)
        contraction = jnp.squeeze(contraction)

        kappa_abc = jnp.expand_dims(Omega**2, axis=((1,2,3))) * contraction
        
        kappa_integrand = jnp.expand_dims(weights / dVol_Omega, axis=((1,2,3))) * kappa_abc
        int_kappa_abc = jnp.mean(kappa_integrand, axis=0)

        return int_kappa_abc

    def yukawa_couplings_batched(self, p, batch_size=16384, kappa_dtype=np.float32):
        n = 0
        kappa = jnp.zeros((self.n_harmonic, self.n_harmonic, self.n_harmonic), kappa_dtype)
        n_chunks = p.shape[0] // batch_size
        data = jnp.array_split(p, n_chunks)
        for t, _p in enumerate(tqdm(data, total=len(data))):
            B = _p.shape[0]
            _kappa = self.yukawa_couplings(_p)
            kappa = kappa = math_utils.online_update_array(kappa, _kappa, n, B)
            n += B
        return kappa

    def connection_form(self, p, params):
        A = self.connection_form_V(p, self.section_metric_network, params)
        return A

    @partial(jax.jit, static_argnums=(0,))
    def curvature_form(self, p, params):
        F = self.curvature_form_V(p, self.section_metric_network, params)
        return F

    def curvature_form_fn(self, p, pb, params):
        F_H0 = self.curvature_form_V(p, self.fubini_study_metric_twist_V_DKLR)
        ddbar_h = self.del_z_del_z_bar(p, self.endomorphism_network, params)
        #ddbar_h = self.del_z_del_z_bar(p, self._endomorphism_network, params)
        #emb = self.embedding_matrix_DKLR(p)
        ddbar_h = jnp.einsum("...iu, ...abuv, ...jv->...abij", pb, ddbar_h, jnp.conjugate(pb))
        return F_H0 + ddbar_h

    @partial(jax.jit, static_argnums=(0,))
    def curvature_correction(self, p, pb, params):
        F_H0 = self.curvature_form_V(p, self.fubini_study_metric_twist_V_DKLR)
        d_correction = curvature.del_bar_z(p, self.exact_piece, False, params,
                                           self.fubini_study_metric_twist_V_DKLR)
        d_correction = jnp.einsum("...abiu, ...ju->...abij", d_correction, jnp.conjugate(pb))
        return F_H0 + d_correction
    
    def trace_free_curvature_correction(self, p, pb, params):
        F = self.curvature_correction(p, pb, params)
        TrF = jnp.einsum("...aaij->...ij", F)
        trace_part = 1./self.rank_V * jnp.einsum("...ij, ...ab->...abij", 
            TrF, jnp.eye(self.rank_V, dtype=TrF.dtype))
        return F - trace_part


    @partial(jax.jit, static_argnums=(0,))
    def curvature_correction_conformal(self, p, pb, params):
        F_H0 = self.curvature_form_V(p, self.H0_conformal_fn)
        d_correction = curvature.del_bar_z(p, self.exact_piece, False, params,
                                           self.H0_conformal_fn)
        d_correction = jnp.einsum("...abiu, ...ju->...abij", d_correction, jnp.conjugate(pb))
        return F_H0 + d_correction

    def exact_piece(self, p, params, H0_metric_fn):
        pb = self.pb_fn(math_utils.to_complex(p))
        h = self.endomorphism_network(p, params)  # h^b_a

        dh = curvature.del_z(p, self.endomorphism_network, False, params)  # h^b_{ai}
        dh = jnp.einsum("...abu, ...iu->...abi", dh, pb)
        A_0 = self.connection_form_V(p, H0_metric_fn)

        _A1 = jnp.einsum("...aci, ...cb->...abi", A_0, h)
        _A2 = jnp.einsum("...cbi, ...ac->...abi", A_0, h)
        holo_cov_der_h = dh + _A1 - _A2
        # exact = jnp.einsum("...ca, ...abi->...cbi", jnp.linalg.inv(h), holo_cov_der_h)
        exact = jnp.linalg.solve(h, holo_cov_der_h.reshape(self.rank_V, -1))
        exact = exact.reshape(self.rank_V, self.rank_V, self.cy_dim)
        return exact

    def _section_basis_V(self, p):
        r"""
        Smooth basis of sections of $V$ expressed in a local frame - typically Z_i^k. 
        """
        p_c = math_utils.to_complex(p)
        patch_idx = jnp.argmax(jnp.abs(p_c)[:self.rank_B])
        linear_monomials = poly_utils.monomial_evaluate_log(p, self.mb1)
        linear_monomials = linear_monomials / p_c[patch_idx]

        n_linear = linear_monomials.shape[0]
        section_matrix = jnp.zeros((self.rank_B-1, n_linear), dtype=self.cdtype)
        section_matrix = jnp.insert(section_matrix, patch_idx, linear_monomials, axis=0)
        embedding_matrix = self.embedding_matrix(p)

        return embedding_matrix @ section_matrix
        # return jnp.conjugate(embedding_matrix) @ section_matrix

    def section_basis_V(self, p):
        r"""
        Smooth basis of sections of $V$ expressed in a local frame - typically Z_i^k. 
        """
        p_c = math_utils.to_complex(p)
        patch_idx = jnp.argmax(jnp.abs(p_c)[:self.rank_B])

        linear_monomials = poly_utils.monomial_evaluate_log(p, self.mb1)
        linear_monomials = linear_monomials# / p_c[patch_idx]
        blocks = [linear_monomials] * self.rank_B
        # rank(B) \times (\dim A_k \times rank(B))
        section_matrix = jax.scipy.linalg.block_diag(*blocks)
        embedding_matrix = self.embedding_matrix(p)

        return embedding_matrix @ section_matrix


    def TrF_correction(self, p, pb, params):
        #F_V = hym._curvature_form_V(p, pb, self.fubini_study_metric_V)
        #ddbar_h = self.del_z_del_z_bar(p, self.endomorphism_network, params)
        #ddbar_h = jnp.einsum("...iu, ...abuv, ...jv->...abij", pb, ddbar_h, jnp.conjugate(pb))
        # F = self.curvature_correction(p, pb, params)
        F = self.curvature_form_fn(p, pb, params)
        # F = self.curvature_form(p, pb, params)
        Tr_eta = jnp.einsum("...aaij->...ij", F)

        return Tr_eta

    def contract_TrF(self, p, pb, params):
        Tr_eta = self.TrF_correction(p, pb, params)
        g_inv = jnp.linalg.inv(self._metric_fn(p))  # \bar{\nu} \mu
        return jnp.einsum("...vu, ...uv->...", g_inv, Tr_eta)


    @partial(jax.jit, static_argnums=(0,))
    def codifferential_TrF(self, p, pb, params):

        g_inv = jnp.linalg.inv(self._metric_fn(p))  # \bar{\nu} \mu
        TrF = self.TrF_correction(p, pb, params)
        del_z_TrF = curvature.del_z(p, self.TrF_correction, False, pb, params)  # [\mu, \bar{\nu}, \kappa]
        del_z_TrF = jnp.einsum("...iju, ...ku->...ijk", del_z_TrF, pb)

        Gamma_holo = curvature.christoffel_symbols_kahler(p, self._metric_fn, pb)  # [a, \kappa, b]
        _cov2 = jnp.einsum('...akb, ...av -> ...bvk', Gamma_holo, TrF)   # [b, \bar{\nu}, \kappa]
        covariant_derivative_eta = del_z_TrF - _cov2
        codiff = -jnp.einsum('...vu, ...bvu->...b', g_inv, covariant_derivative_eta)
        return codiff


    @partial(jax.jit, static_argnums=(0,))
    def objective_function(self, data, params):
        #(p, pb, w) = data
        #vol_Omega = jnp.mean(w)
        #codiff = vmap(self.codifferential_TrF, in_axes=(0,0,None))(p, pb, params)
        #codiff = jnp.squeeze(codiff)
        # g_pred = vmap(self.metric_fn)(p)
        #loss = jnp.mean(jnp.abs(codiff) * jnp.expand_dims(w, axis=1)) / vol_Omega
        loss = hym.objective_function_implicit_slope_V(data, params, 
                                                       self.trace_free_curvature_correction,
                                                       self._metric_fn, self.section_metric_network, 
                                                       self.rank_V)
        return loss

    def conformal_rescale_network(self, p, params):
        f = models.phi_head(p, params, self.n_hyper, tuple(self.ambient), activation=self.activation)
        return f

    def section_metric_network(self, p, params):
        r"""
        Returns a smooth section of $Sym(V^* \otimes V^*$) from basis of sections for $V$.
        """
        """
        C = 1.
        p_c = math_utils.to_complex(p)
        H_fs_V = self.fubini_study_metric_V(p)
        
        # (n_h, n_Vk, n_Ok) * n_A if all ambient space factors identical
        # TODO
        k = 1
        coeffs = models.coeff_head_holoV(p, params, self.n_homo_coords, tuple(self.ambient), self.N_sb, 
                                    k*self.N_sb, None, self.n_harmonic, complex_kernel=True, activation=activation)
        coeffs = jnp.squeeze(coeffs[0])
        M = coeffs @ self.dagger(coeffs)
        
        # Overcomplete basis of V-sections
        sv = self.section_basis_V(p)  # [dim, big_number]
        dual_sv = jnp.einsum("...ab, ...bn->...an", H_fs_V, jnp.conjugate(sv))
        H = jnp.einsum("...am, ...mn, ...bn->...ab", dual_sv, M, jnp.conjugate(dual_sv))
        return H_fs_V + H

        H_inv = jnp.einsum("...bm, ...mn, ...an->...ba", jnp.conjugate(sv), M, sv)  # [\bar{b}, a]
        return jnp.linalg.inv(H_inv)  # [a, \bar{b}]
        """

        coeffs = models.cholesky_head(p, params, self.n_homo_coords, tuple(self.ambient), self._N_sb)
        # tsb = self.twisted_section_basis(p)
        tsb = self.twisted_section_basis_DKLR(p)
        G_inv = jnp.einsum("...am, ...mn, ...bn->...ab", jnp.conj(tsb), coeffs, tsb)
        return jnp.linalg.inv(G_inv)

    def conformal_change(self, p, params):
        pb = self.pb_fn(math_utils.to_complex(p))
        xi = self.TrF_H_0(p)
        # xi = curvature.ricci_form_kahler(p, self.fs_metric_fn, pb)
        ddbar_f = self.del_z_del_z_bar(p, self.conformal_rescale_network, params)
        ddbar_f = jnp.einsum("...iu, ...uv, ...jv->...ij", pb, ddbar_f, jnp.conjugate(pb))
        return xi + self.rank_V * ddbar_f

    @partial(jax.jit, static_argnums=(0,))
    def codifferential_TrF_conformal(self, p, pb, params):

        g_inv = jnp.linalg.inv(self._metric_fn(p))  # \bar{\nu} \mu
        TrF = self.conformal_change(p, params)
        del_z_TrF = curvature.del_z(p, self.conformal_change, False, params)  # [\mu, \bar{\nu}, \kappa]
        del_z_TrF = jnp.einsum("...iju, ...ku->...ijk", del_z_TrF, pb)

        Gamma_holo = curvature.christoffel_symbols_kahler(p, self._metric_fn, pb)  # [a, \kappa, b]
        _cov2 = jnp.einsum('...akb, ...av -> ...bvk', Gamma_holo, TrF)   # [b, \bar{\nu}, \kappa]
        covariant_derivative_eta = del_z_TrF - _cov2
        codiff = jnp.einsum('...vu, ...bvu->...b', g_inv, covariant_derivative_eta)
        return -codiff

    @partial(jax.jit, static_argnums=(0,))
    def objective_function_conformal(self, data, params, MAX_NORM=20.):
        (p, pb, w) = data
        vol_Omega = jnp.mean(w)
        codiff = vmap(self.codifferential_TrF_conformal, in_axes=(0,0,None))(p, pb, params)
        codiff = jnp.squeeze(codiff)  # [..., i]

        # g_pred = vmap(self._metric_fn)(p)
        # g_inv = jnp.linalg.inv(g_pred)
        # abs_codiff = jnp.real(
        #     jnp.einsum("...vu, ...u, ...v", g_inv, codiff, jnp.conj(codiff)))
        abs_codiff = jnp.mean(jnp.abs(codiff), axis=-1)
        abs_codiff = jnp.where(abs_codiff < MAX_NORM, abs_codiff, 0.)
        
        loss = jnp.mean(abs_codiff * w) / vol_Omega
        #loss = hym.objective_function_implicit_slope(data, params,
        #        self.conformal_change, self._metric_fn)
        return loss

    def _contract_TrF(self, p, params):
        eta = self.conformal_change(p, params)
        g_inv = jnp.linalg.inv(self._metric_fn(p))  # \bar{\nu} \mu
        return jnp.einsum("...vu, ...uv->...", g_inv, eta)

    def _contract_TrF_H0(self, p):
        eta = self.TrF_H_0(p)
        g_inv = jnp.linalg.inv(self._metric_fn(p))  # \bar{\nu} \mu
        return jnp.einsum("...vu, ...uv->...", g_inv, eta)

    @partial(jax.jit, static_argnums=(0,))
    def _codifferential_TrF_conformal(self, p, pb, params):

        del_z_contraction = curvature.del_z(p, self._contract_TrF, False, params)
        codiff = jnp.einsum("...u, ...iu->...i", del_z_contraction, pb)
        return codiff

    def TrF_H_0(self, p):
        F_H_0 = self.curvature_form_V(p, self.fubini_study_metric_twist_V_DKLR)
        return jnp.einsum("...aaij->...ij", F_H_0)
    
    def ddbar_log_det_H_0(self, p, pb):
        hess = self.del_z_del_z_bar(p, self.log_det_H_0)
        return jnp.einsum("...iu, ...uv, ...jv->...ij", pb, hess, jnp.conjugate(pb))

    def log_det_H_0(self, p):
        H_0 = self.fubini_study_metric_twist_V_DKLR(p)
        s, logdet = jnp.linalg.slogdet(H_0)
        return logdet + 1j * jnp.pi * (s < 0)

    def H0_conformal_change(self, p, params):
        f = self.conformal_rescale_network(p, params)
        H0 = self.fubini_study_metric_twist_V_DKLR(p)
        return jnp.expand_dims(jnp.exp(f), (0,1)) * H0

    def embedding_matrix_DKLR(self, p, patch_idx=None):
        r"""
        Describes embedding $\iota: V \righthookarrow B$.
        """
        if patch_idx is None:
            patch_idx = jnp.argmax(jnp.abs(math_utils.to_complex(p))[:self.rank_B])
        # patch_idx = 0
        proj = jnp.eye(self.rank_V, dtype=self.cdtype)
        f_p = poly_utils.monomial_evaluate_log(p, self.monad_map_power_matrix_DKLR)
        col = -f_p / f_p[patch_idx]  # f_p[patch_idx] should usually be 1.
        col = jnp.delete(col, patch_idx, assume_unique_indices=True)
        return jnp.insert(proj, patch_idx, col, axis=-1)

    def twisted_section_basis_DKLR(self, p, ambient=False):
        r"""
        Holomorphic sections of twisted bundle $V \otimes O_X(k)$,
        expressed in a local frame - typically Z_i^k. 
        """
        #p_c = math_utils.to_complex(p)
        #p = math_utils.to_real(p_c)
        patch_idx = jnp.argmax(jnp.abs(math_utils.to_complex(p))[:self.rank_B])
        # patch_idx = 0
        Ok_powers = self.mb1
        Ok_monomials = poly_utils.monomial_evaluate_log(p, Ok_powers)
        blocks = [Ok_monomials] * self.rank_B
        section_matrix = jax.scipy.linalg.block_diag(*blocks)
        section_matrix = jnp.delete(section_matrix, 
                                    self.n_linear * patch_idx + patch_idx, axis=-1, 
                                    assume_unique_indices=True)
        if ambient is True: return section_matrix
        embedding_matrix = self.embedding_matrix_DKLR(p, patch_idx)
        return embedding_matrix @ section_matrix

    def fubini_study_metric_twist_V_DKLR(self, p):
        tsb = self.twisted_section_basis_DKLR(p)
        fs_inv = jnp.einsum("...am, ...bm->...ba", tsb, jnp.conjugate(tsb))
        fs = jnp.linalg.inv(fs_inv)
        return fs

    def endomorphism_network(self, p, params, normalise_det=False):
        r"""
        Model a section of the endomorphism bundle on $V$ as a matrix of coefficients (each of which is 
        a global function), which parameterise the section via a linear combination of a section of 
        $V$ tensored by a dual section.
        $$ h^b_a = \sum_{mn} H^{mn} S^b_m \otimes \hat{S}_{an}~. $$
        """
        f = self.conformal_fn(p)
        h0 = jnp.eye(self.rank_V, dtype=self.cdtype)
        normalise_det = True
        # TODO
        coeffs = models.cholesky_head(p, params, self.n_homo_coords, tuple(self.ambient), 
                                      self._N_sb, normalise_det=False)
        tsb = self.twisted_section_basis_DKLR(p)
        H_fs_V = self.fubini_study_metric_twist_V_DKLR(p)

        tsb_dual = jnp.einsum("...ab, ...bm->...am", H_fs_V, jnp.conjugate(tsb))
        h = jnp.einsum("...am, ...mn, ...bn->...ab", tsb, coeffs, tsb_dual)
        h = h + h0
        if normalise_det is True:
            _, logdet = jnp.linalg.slogdet(h)
            scale = jnp.exp(-logdet / self.rank_V)
            h = scale * h
        return h * jnp.exp(f)

    @staticmethod
    def int_dVol_Omega(f, w, vol_w):
        return jnp.mean(f * w) / vol_w

    def loss_breakdown(self, data, params):
        
        loss = self.objective_function(data, params)
        p, pbs, w = data

        # h = vmap(self.section_metric_network, in_axes=(0,None))(p, params)
        h = vmap(self.endomorphism_network, in_axes=(0,None))(p, params)
        g = vmap(self._metric_fn)(p)
        # F = vmap(self.curvature_form, in_axes=(0,0,None))(p, pbs, params)
        # F = vmap(self.curvature_form_fn, in_axes=(0,0,None))(p, pbs, params)
        # F = vmap(self.curvature_correction, in_axes=(0,0,None))(p, pbs, params)
        F = vmap(self.trace_free_curvature_correction, in_axes=(0,0,None))(p,
                pbs, params)

        g_tr_F = jnp.einsum("...vu, ...abuv->...ab", jnp.linalg.inv(g), F)
        det_g_tr_F = jnp.linalg.det(g_tr_F)
        max_eig = vmap(jnp.linalg.norm)(g_tr_F)
        vol_Omega = jnp.mean(w)
        g_tr_F = vmap(jnp.trace)(g_tr_F)

        return {'loss': loss, 'g_tr_F': jnp.mean(w * g_tr_F) / vol_Omega, "max_eig": jnp.mean(w * max_eig) / vol_Omega,
                'det_F_g': jnp.mean(w * det_g_tr_F) / vol_Omega, "det_h": jnp.mean(w * jnp.linalg.det(h)) / vol_Omega,
                'Tr_F_g_var': jnp.var(jnp.abs(g_tr_F))}

    def loss_breakdown_conformal(self, data, params):
        
        loss = self.objective_function_conformal(data, params)
        p, pb, w = data
        f = vmap(self.conformal_rescale_network, in_axes=(0,None))(p, params)
        H0 = vmap(self.fubini_study_metric_twist_V_DKLR)(p)
        H = jnp.expand_dims(jnp.exp(f), (1,2)) * H0
        g = vmap(self._metric_fn)(p)
        g_inv = jnp.linalg.inv(g)

        codiff = vmap(self._codifferential_TrF_conformal, in_axes=(0,0,None))(p, pb, params)
        codiff_norm = jnp.mean(jnp.abs(codiff) * jnp.expand_dims(w, axis=1))
        # codiff_norm = jnp.einsum("...vu, ...u, ...v", g_inv[:k], codiff, jnp.conj(codiff))
        var = hym.objective_function_implicit_slope(data, params, self.conformal_change, self._metric_fn)

        TrF = vmap(self.conformal_change, in_axes=(0,None))(p, params)
        TrF_H0 = vmap(self.TrF_H_0)(p)
        Lambda_TrF = jnp.einsum("...vu, ...uv->... ", g_inv, TrF)
        Lambda_TrF_H0 = jnp.einsum("...vu, ...uv->... ", g_inv, TrF_H0)
        vol_Omega = jnp.mean(w)

        return {'loss': loss, 'f avg': jnp.mean(w * f) / vol_Omega, "Λ Tr F": jnp.mean(w * Lambda_TrF) / vol_Omega,
                'Λ Tr F_H0': jnp.mean(w * Lambda_TrF_H0) / vol_Omega, "det_H": jnp.mean(w * jnp.linalg.det(H)) / vol_Omega,
                '|∂† Tr F|^2': codiff_norm / vol_Omega, 'Var[Λ Tr F]': var}

    def callback(self, val_data, params, storage, logger, epoch, t0,
                 slope: float = None, conformal_train=False):
        
        if conformal_train is True:
            loss_breakdown_dict = self.loss_breakdown_conformal(val_data, params)
        else:
            loss_breakdown_dict = self.loss_breakdown(val_data, params)

        loss_breakdown_dict = jax.device_get(loss_breakdown_dict)
        summary = jax.tree_util.tree_map(lambda x: x.item(), loss_breakdown_dict)

        mode = 'VAL'
        logs = [f"{k}: {v:.4f}" for (k,v) in summary.items()]        
        logger.info(f"[{time.time()-t0:.1f}s]: [{mode}] | Epoch: {epoch}" + ''.join([f" | {log}" for log in logs]))

        [storage[k].append(v) for (k,v) in summary.items()]
        if epoch % self.save_interval == 0:
            utils.save_logs(storage, self.name, epoch)
        return storage
    
    @staticmethod
    def _create_train_state(rng, model, optimizer):
        rng, init_rng = random.split(rng)
        # params = model.init(rng, jnp.ones([1, data_dim]))['params']
        params = model.init(rng)['params']
        opt_state = optimizer.init(params)
        return params, opt_state, init_rng
    
    def fit(self, data_path, epochs: int = 32, batch_size: int = 512, lr: float = 1e-4,
            conformal_fn = None, shuffle_rng = np.random.default_rng(), name = None):
        from datetime import datetime

        self.name = f"HYM_{datetime.now().strftime('%Y-%m-%d_%H')}" if name is None else name
        self.eval_interval = 1  # epochs
        self.save_interval = 8
        self.eval_interval_t = 512  # iterations

        storage = defaultdict(list)
        logger = utils.logger_setup(self.name, filepath=os.path.abspath(__file__))
        data_path = os.path.join(data_path, 'dataset.npz')
        os.makedirs(os.path.join("experiments", self.name), exist_ok=True)
        logger.info(f'Dataset: {data_path}')

        A_train, A_val, train_loader, val_loader, psi = dataloading.initialize_loaders_train(shuffle_rng, data_path, 
            batch_size, logger=logger)
        dataset_size = A_train[0].shape[0]

        # Normalize slope
        vol = jnp.mean(A_train[1])
        if self._slope is not None: self._slope *= vol

        try:
            device = jax.devices('gpu')[0]
        except:
            print("gpu not detected, falling back to cpu.")
            device = jax.devices('cpu')[0]

        # optimisation stuff - separate later
        key = jax.random.key(42)
        key, _k = jax.random.split(key)

        grad_threshold = 1.
        _tx = optax.chain(
          optax.clip(grad_threshold),
          optax.adamw(learning_rate=lr),
        )
        self.n_units_harmonic = [48,48,48]
        if conformal_fn is not None: self.conformal_fn = conformal_fn
        # model_class = models.CoeffNetwork_spectral_nn_CICY_holoV
        # k = 1
        # bundle_metric_model = model_class(self.n_homo_coords, self.ambient, self.n_units_harmonic, n_1=self.N_sb,
        #                                   n_2=k * self.N_sb, n_harmonic=self.n_harmonic, complex_kernel=True,
        #                                   activation=nn.gelu)
        coeff_class = models.CholeskyNetwork
        bundle_metric_model = coeff_class(self.n_homo_coords, self.ambient, self.n_units_harmonic, 
                matrix_dim=self._N_sb)

        _params, _opt_state, _ = create_train_state(_k, bundle_metric_model, _tx, data_dim=self.n_homo_coords * 2)
        # _params, _opt_state, _ = self._create_train_state(_k, bundle_metric_model, _tx)
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(_params))
        logger.info(f'Params (Count: {param_count})=========>>>')
        logger.info(jax.tree_util.tree_map(lambda x: x.shape, _params))
        logger.info(bundle_metric_model.tabulate(_k, jnp.ones([1, self.n_homo_coords * 2])))

        t0 = time.time()
        with jax.default_device(device):
            for epoch in range(epochs):

                if epoch % self.eval_interval == 0: 
                    val_loader, val_data = dataloading.get_validation_data(val_loader, batch_size, A_val, shuffle_rng)
                    p, w, _ = val_data
                    pb = vmap(self.pb_fn)(math_utils.to_complex(p))
                    val_data = (p, pb, w)
                    storage = self.callback(
                        val_data, _params, storage, logger, epoch, t0, self._slope)

                if epoch > 0: 
                    train_loader = dataloading.data_loader(A_train, batch_size, shuffle_rng)

                wrapped_train_loader = tqdm(train_loader, desc=f'Epoch {epoch}', total=dataset_size//batch_size, 
                                            colour='green', mininterval=0.1)

                for t, data in enumerate(wrapped_train_loader):
                    p, w, _ = data
                    pb = vmap(self.pb_fn)(math_utils.to_complex(p))
                    data = (p, pb, w)

                    _params, _opt_state, loss = hym._train_step(data, _params, _opt_state, _tx, self.objective_function)
                    wrapped_train_loader.set_postfix_str(f"loss: {loss:.5f}", refresh=False)

                    if t % self.eval_interval_t == 0:
                        storage["train_loss"].append(loss.item())

                if epoch % self.save_interval == 0:
                    utils.basic_ckpt(_params, _opt_state, self.name, f'{epoch}')

        utils.basic_ckpt(_params, _opt_state, self.name, 'FIN')
        utils.save_logs(storage, self.name, 'FIN')
        return _params, storage


    def fit_conformal(self, data_path, epochs: int = 32, batch_size: int = 512, lr: float = 1e-4,
            shuffle_rng = np.random.default_rng(), name=None):
        from datetime import datetime

        self.name = f"HYM_conformal_{datetime.now().strftime('%Y-%m-%d_%H')}" if name is None else name
        self.eval_interval = 1  # epochs
        self.save_interval = 8
        self.eval_interval_t = 512  # iterations

        storage = defaultdict(list)
        logger = utils.logger_setup(self.name, filepath=os.path.abspath(__file__))
        data_path = os.path.join(data_path, 'dataset.npz')
        os.makedirs(os.path.join("experiments", self.name), exist_ok=True)
        logger.info(f'Dataset: {data_path}')

        A_train, A_val, train_loader, val_loader, psi = dataloading.initialize_loaders_train(shuffle_rng, data_path, 
            batch_size, logger=logger)
        dataset_size = A_train[0].shape[0]

        # Normalize slope
        vol = jnp.mean(A_train[1])
        if self._slope is not None: self._slope *= vol

        try:
            device = jax.devices('gpu')[0]
        except:
            print("gpu not detected, falling back to cpu.")
            device = jax.devices('cpu')[0]

        # optimisation stuff - separate later
        key = jax.random.key(42)
        key, _k = jax.random.split(key)

        grad_threshold = 1.
        _tx = optax.chain(
          optax.clip(grad_threshold),
          optax.adamw(learning_rate=lr),
        )
        self.n_units = [48,48,48] 
        model_class = models.LearnedVector_spectral_nn
        model = model_class(self.n_homo_coords, self.ambient, self.n_units)
        _params, _opt_state, _ = create_train_state(_k, model, _tx, data_dim=self.n_homo_coords * 2)
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(_params))
        logger.info(f'Params (Count: {param_count})=========>>>')
        logger.info(jax.tree_util.tree_map(lambda x: x.shape, _params))
        logger.info(model.tabulate(_k, jnp.ones([1, self.n_homo_coords * 2])))

        t0 = time.time()
        with jax.default_device(device):
            for epoch in range(epochs):

                if epoch % self.eval_interval == 0: 
                    val_loader, val_data = dataloading.get_validation_data(val_loader, batch_size, A_val, shuffle_rng)
                    p, w, _ = val_data
                    pb = vmap(self.pb_fn)(math_utils.to_complex(p))
                    val_data = (p, pb, w)
                    storage = self.callback(
                        val_data, _params, storage, logger, epoch, t0, self._slope, conformal_train=True)

                if epoch > 0: 
                    train_loader = dataloading.data_loader(A_train, batch_size, shuffle_rng)

                wrapped_train_loader = tqdm(train_loader, desc=f'Epoch {epoch}', total=dataset_size//batch_size, 
                                            colour='green', mininterval=0.1)

                for t, data in enumerate(wrapped_train_loader):
                    p, w, _ = data
                    pb = vmap(self.pb_fn)(math_utils.to_complex(p))
                    data = (p, pb, w)

                    _params, _opt_state, loss = hym._train_step(data, _params, _opt_state, _tx, 
                                                                self.objective_function_conformal)
                    wrapped_train_loader.set_postfix_str(f"loss: {loss:.5f}", refresh=False)

                    if t % self.eval_interval_t == 0:
                        storage["train_loss"].append(loss.item())

                if epoch % self.save_interval == 0:
                    utils.basic_ckpt(_params, _opt_state, self.name, f'{epoch}')

        utils.basic_ckpt(_params, _opt_state, self.name, 'FIN')
        utils.save_logs(storage, self.name, 'FIN')
        return _params, storage

    def sample_intersect_hypersurface(self, key: random.PRNGKey, n_p: int, 
                                      LOCUS_TOL: float = 1e-10):

        """Samples from manifold defined as a hypersurface in projective space
        by solving for the intersection 'Q(p + t * q)'.
        """

        _key, key = random.split(key, 2)

        # Generate points on S^{2n+1} (S^{2n+1}/U(1) \cong CP^n)
        c_dim = self.cy_dim + 2  # homo. coords plus hypersurface constraint
        n_intersect = np.ceil(n_p / c_dim).astype(int)
        sphere_pts = pointgen.S2np1_uniform(_key, 2*n_intersect, self.cy_dim+1)
        p, q = jnp.split(sphere_pts, 2)

        # solve for intersection of line with hypersurface, compute 
        # Q(p + t * q), find coefficients of terms of each power symbolically
        t_coeffs_data, generators = pointgen.univariate_coefficient_data(self.cy_dim, self.monomials, self.coefficients)
        
        # find coeffs and pass to root solver - TODO: extend to gpu.
        # on cpu because `linalg.eig` not supported on gpu.
        pts, t_coeffs = vmap(pointgen.root_solver, in_axes=(0,0,None,None,None))(
            p, q, t_coeffs_data.values(), tuple(generators), jax.devices()[0])
        pts = pts.reshape(-1, c_dim)

        # recall Bezout's theorem guarantees `c_dim` intersecting points
        # rescale points - return homogeneous coords with $\max{|z_i|} = 1$
        pts, *_ = math_utils.rescale(pts.reshape(-1, c_dim)[:n_p])

        return pts

    def fibre_metric_from_H(self, p, H, aux=False):
        S = self.twisted_section_basis_DKLR(p)
        S_c = jnp.conjugate(S)
        G_inv = jnp.einsum("mn, ...am, ...bn->...ab", H, S, S_c)
        G = jnp.einsum("...ba->...ab", jnp.linalg.inv(G_inv))  # G_{a \overline{b}}
        if aux is True: return G, S, S_c
        return G

    @partial(jax.jit, static_argnums=(0,))
    def curvature_form_donaldson(self, p, H):
        F = self.curvature_form_V(p, self.fibre_metric_from_H, H)
        return F

    def donaldson_step(self, key, H, n_batches, batch_size):
        # Single Donaldson step using the entire dataset
        T = jnp.zeros((self._N_sb, self._N_sb), dtype=jnp.complex128)
        init_val = (T, 0.)
        keys = jax.random.split(key, n_batches)

        def _batch_step(i, carry):
            _T, _vol_Omega = carry
            p = self.sample_intersect_hypersurface(keys[i], batch_size)
            abs_poly_val = jnp.abs(vmap(alg_geo.evaluate_poly, in_axes=(0,None,None))(p, 
                            self.monomials, self.coefficients))
            w, _pb, _dVol_Omega, *_ = vmap(alg_geo.compute_integration_weights, in_axes=(0,None,None,None))( 
                p, self.dQdz_monomials, self.dQdz_coeffs, self.cy_dim)
            delta_T, vol_Omega_batch = self._G_update(p, H, w)

            T = math_utils.online_update_array(_T, delta_T, i*batch_size, batch_size)
            vol_Omega = math_utils.online_update(_vol_Omega, vol_Omega_batch, i*batch_size, batch_size)

            return T, vol_Omega

        T, vol_Omega = jax.lax.fori_loop(0, n_batches, _batch_step, init_val)
        T = T * H.shape[0] / vol_Omega / self.rank_V

        # inverse of output of T-operator is the Hermitian matrix on the space of sections
        H_update = jnp.linalg.inv(T).T
        return H_update
            
    @partial(jax.jit, static_argnums=(0,))
    def _G_update(self, p, H, w):
        G, S, S_c = vmap(self.fibre_metric_from_H, in_axes=(0,None,None))(p, H, True)
        integrand = jnp.einsum("...am, ...ab, ...bn->...mn", S, G, S_c)
        
        delta_T = jnp.mean(integrand * jnp.expand_dims(w, axis=(1,2)), axis=0)
        vol_Omega_batch = jnp.mean(w)
        return delta_T, vol_Omega_batch


    @staticmethod
    def check_min(p, patch, threshold=1e-1):
        mask = jnp.abs(p)[:,patch] > threshold
        return math_utils.rescale_patch(p[mask], patch)


    def _donaldson_step(self, key, H, current_i, n_batches, batch_size, LOCUS_TOL=1e-10):
        T = jnp.zeros((self._N_sb, self._N_sb), dtype=jnp.complex128)
        vol_Omega = 0.0
        keys = jax.random.split(key, n_batches)

        # Wrap the inner loop with tqdm, using leave=False so it disappears when done.
        batch_progress = tqdm(range(n_batches), 
                              desc=f"    ↳ Iter {current_i+1} batches", 
                              leave=False,
                              colour='blue')

        n = 0
        for i in batch_progress:
            p = self.sample_intersect_hypersurface(keys[i], batch_size)
            abs_poly_val = jnp.abs(vmap(alg_geo.evaluate_poly, in_axes=(0,None,None))(p,
                            self.monomials, self.coefficients))
            p = p[abs_poly_val < LOCUS_TOL]
            p = self.check_min(p, patch=2)  # move to patch 0
            B = p.shape[0]
            w, _pb, _dVol_Omega, *_ = vmap(alg_geo.compute_integration_weights, in_axes=(0,None,None,None))(
                p, self.dQdz_monomials, self.dQdz_coeffs, self.cy_dim)

            delta_T, vol_Omega_batch = self._G_update(p, H, w)

            T = math_utils.online_update_array(T, delta_T, n, B)
            vol_Omega = math_utils.online_update(vol_Omega, vol_Omega_batch, n, B)
            n += B

        T = T * H.shape[0] / vol_Omega / self.rank_V

        # inverse of output of T-operator is the Hermitian matrix on the space of sections
        H_update = jnp.linalg.inv(T).T
        return H_update


    def generalised_donaldson(self, iterations=16, batch_size=8192):

        from cymyc.utils import pointgen
        key = jax.random.PRNGKey(42)
        Np = (10 * self._N_sb**2 + 50000) * 25
        print(f"Using {Np} points ...")
        print(f"Using device {jax.devices()} ...")

        n_batches = Np // batch_size + 1
        H0 = jnp.eye(self._N_sb, dtype=jnp.complex128)
        #step = jax.jit(partial(self._donaldson_step, n_batches=n_batches, 
        #                 batch_size=batch_size))
        step = partial(self._donaldson_step, n_batches=n_batches, batch_size=batch_size)

        i, H = 0, H0  # convention: H^{m \overline{n}}

        progress = tqdm(range(iterations), desc='Gen. Donaldson iterations', total=iterations,
                colour='green')

        eps = 1e-10
        for i in progress:
            H_prev = H
            H = (H + self.dagger(H)) / 2
            H = step(key, H, i)
            det_H = jnp.linalg.det(H)
            # H = H / ((det_H + eps)**(1.0 / H.shape[0]))
            H = H / jnp.max(jnp.abs(H))
            diff = jnp.linalg.norm(H - H_prev) / jnp.linalg.norm(H_prev)
            norm_H = jnp.linalg.norm(H)

            metrics_to_log = {
                "Rel. Change": jnp.linalg.norm(H - H_prev) / jnp.linalg.norm(H_prev),
                "Norm H": norm_H,
                "det H": jnp.linalg.det(H),
            }

            log_parts = [f"{name}: {value.item():.4e}" for name, value in metrics_to_log.items()]
            summary_str = f"Iter {i+1}/{iterations} -> " + ", ".join(log_parts)

            tqdm.write(summary_str)

            progress.set_postfix({
                "Relative change": f"{diff.item():.4e}",
            }, refresh=False)
            key, _ = jax.random.split(key)

        return H
