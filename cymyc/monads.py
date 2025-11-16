import jax
# jax.config.update("jax_enable_x64", True)

import numpy as np  # original CPU-backed NumPy
import jax.numpy as jnp

from jax import jit, jacfwd, vmap, random
import optax

from functools import partial
from datetime import datetime

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
        # $ 0 \rightarrow A \rightarrow B \rightarrow V \rightarrow 0 $
        # CHANGE THIS
        self.rank_V = 2
        self.twisting_degree = 3  # 2 for ABKO, 1 for DKLR, 3 for AG
        self.line_bundle_B = (1,1,1)    # (1,1,1,1)
        self.line_bundle_A_twist = 1
        self.default_idx = 0


        self.rank_B = len(self.line_bundle_B)
        self.n_frames = self.rank_B
        self.cdtype = np.complex64
        self.mb1 = jnp.asarray(poly_utils.monomial_basis(ambient, 1))
        self.mb2 = jnp.asarray(poly_utils.monomial_basis(ambient, 2))
        self.mb3 = jnp.asarray(poly_utils.monomial_basis(ambient, 3))
        self.mb4 = jnp.asarray(poly_utils.monomial_basis(ambient, 4))
        self.mb5 = jnp.asarray(poly_utils.monomial_basis(ambient, 5))
        self.mb6 = jnp.asarray(poly_utils.monomial_basis(ambient, 6))
        self.degree_to_monomial_basis = {1: self.mb1, 2: self.mb2, 3: self.mb3, 4: self.mb4, 5: self.mb5,
                6: self.mb6}
        self.n_linear = len(self.mb1)

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
            self.monomial_basis = poly_utils.MonomialBasis(ambient, self.twisting_degree + self.line_bundle_B[0])
        else:
            self.monomial_basis = poly_utils.MonomialBasisReduced(ambient, self.twisting_degree + self.line_bundle_B[0], defining_polys)

        self.all_mono_eval_fn = jax.tree_util.Partial(poly_utils.monomial_evaluate_log, 
                                                      s_k=self.monomial_basis.power_matrix, 
                                                      conj=False)
        
        variables = sp.symarray('z', ambient.item() + len(ambient))
        _monad_map_AG = [v**3 for v in variables[:4]]
        self.monad_map_power_matrix_AG = poly_utils.monomials_to_power_matrix(_monad_map_AG, variables)
        
        self.eps_3d = jnp.array(math_utils.n_dim_eps_symbol(3))
        self.activation = nn.gelu

        _monad_map_DKLR = [v for v in variables[:4]]
        self.monad_map_power_matrix_DKLR = poly_utils.monomials_to_power_matrix(_monad_map_DKLR, variables)

        _monad_map_ABKO = [v**2 for v in variables[:3]]
        self.monad_map_power_matrix_ABKO = poly_utils.monomials_to_power_matrix(_monad_map_ABKO, variables)

        # CHANGE THIS
        self.monad_map_power_matrix = self.monad_map_power_matrix_ABKO  # DKLR
        #self.monad_map_power_matrix = self.monad_map_power_matrix_DKLR
        # self.monad_map_power_matrix = self.monad_map_power_matrix_AG

        monad_map_degree = int(self.monad_map_power_matrix.max())
        # quotient out by polynomials in the subspace bundle
        self.n_quotient = math.comb(self.ambient_dim + self.twisting_degree - monad_map_degree, self.twisting_degree - monad_map_degree)
        self._N_sb = len(self.degree_to_monomial_basis[self.twisting_degree]) * self.rank_B - self.n_quotient
        self.lr_approx = min(0, self._N_sb)  # set to zero for full dense matrix

        self.conf_mat, p_conf_mat = math_utils._configuration_matrix([monomials], ambient)
        self.t_degrees = math_utils._find_degrees(self.ambient, self.n_hyper, self.conf_mat)
        self.kmoduli_ambient = math_utils._kahler_moduli_ambient_factors(self.cy_dim, self.ambient, self.t_degrees)
        self.n_units = [48,48,48]
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
        # A = jnp.einsum("...bc, ...abi->...cai", H_inv, del_H)  # A^c_{ai}
        A = jnp.einsum("...aci, ...cb->...abi", del_H, H_inv)  # A_{ai}^b
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

    @staticmethod
    def _to_real_vec(z):
        # complex vector -> real 2n vector (Re, Im)
        return jnp.concatenate([jnp.real(z), jnp.imag(z)], axis=-1)

    @staticmethod
    def _J_op(v):
        # Complex structure on R^{2n}: J(x,y)=(-y,x)
        n = v.shape[-1] // 2
        x, y = v[..., :n], v[..., n:]
        return jnp.concatenate([-y, x], axis=-1)

    @staticmethod
    def _hess_dir2(f_scalar, x, u, v):
        # Second directional derivative H[u,v] via nested JVPs; f_scalar: R^{2n} -> R
        return jax.jvp(lambda xx: jax.jvp(f_scalar, (xx,), (u,))[1], (x,), (v,))[1]

    def _ddbar_scalar_hvp_in_pb(self, p, pb, f_scalar):
        """
        Compute ∂∂̄ f at p in the holomorphic frame given by pb using HVPs only.
        Returns a complex (n x n) matrix.
        """
        x = p  # real coords (2n,)
        U = vmap(self._to_real_vec)(pb)     # [n, 2n]
        JU = vmap(self._J_op)(U)

        def ddbar_ij(ui, vi, jui, jvi):
            H_uv   = self._hess_dir2(f_scalar, x, ui,  vi)
            H_JuJv = self._hess_dir2(f_scalar, x, jui, jvi)
            H_Juv  = self._hess_dir2(f_scalar, x, jui, vi)
            H_uJv  = self._hess_dir2(f_scalar, x, ui,  jvi)
            real_part = 0.25 * (H_uv + H_JuJv)
            imag_part = -0.25 * (H_Juv - H_uJv)
            return real_part + 1j * imag_part

        ddbar = vmap(
            lambda ui, jui: vmap(lambda vi, jvi: ddbar_ij(ui, vi, jui, jvi))(U, JU)
        )(U, JU)  # [n, n] complex
        return ddbar

    def _logdet_H_scalar(self, p, endo_params):
        # log det H via Cholesky for stability
        H = self.section_metric_network(p, endo_params)
        L = jnp.linalg.cholesky(H)
        return 2.0 * jnp.sum(jnp.log(jnp.real(jnp.diag(L))))

    def _logdet_H0_scalar(self, p):
        H0 = self.fubini_study_metric_twist_V(p)
        L = jnp.linalg.cholesky(H0)
        return 2.0 * jnp.sum(jnp.log(jnp.real(jnp.diag(L))))

    @partial(jax.jit, static_argnums=(0,))
    def ddbar_log_det_H_hvp(self, p, pb, endo_params):
        f_scalar = lambda x: self._logdet_H_scalar(x, endo_params)
        return self._ddbar_scalar_hvp_in_pb(p, pb, f_scalar)

    @partial(jax.jit, static_argnums=(0,))
    def ddbar_log_det_H0_hvp(self, p, pb):
        f_scalar = lambda x: self._logdet_H0_scalar(x)
        return self._ddbar_scalar_hvp_in_pb(p, pb, f_scalar)

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

    def connection_form(self, p, params):
        A = self.connection_form_V(p, self.section_metric_network, params)
        return A

    @partial(jax.jit, static_argnums=(0,))
    def curvature_form(self, p, params):
        F = self.curvature_form_V(p, self.section_metric_network, params)
        return F

    @partial(jax.jit, static_argnums=(0,))
    def _curvature_form(self, p, params, conf_params=None, frame_idx=None):
        H_fn = jax.tree_util.Partial(self.section_metric_network, conf_params=conf_params,
                conformal_factor=True, frame_idx=frame_idx)
        F = self.curvature_form_V(p, H_fn, params)
        return F


    def curvature_form_fn(self, p, pb, params):
        F_H0 = self.curvature_form_V(p, self.fubini_study_metric_twist_V)
        ddbar_h = self.del_z_del_z_bar(p, self.endomorphism_network, params)
        ddbar_h = jnp.einsum("...iu, ...abuv, ...jv->...abij", pb, ddbar_h, jnp.conjugate(pb))
        return F_H0 + ddbar_h

    @partial(jax.jit, static_argnums=(0,))
    def curvature_correction(self, p, pb, params, conf_params=None, frame_idx=None):
        #return self._curvature_form(p, params, conf_params, frame_idx)
        H_fn_i = jax.tree_util.Partial(self.fubini_study_metric_twist_V, frame_idx=frame_idx, transport=False)
        F_H0 = self.curvature_form_V(p, H_fn_i)
        # F_H0 = self.curvature_form_V(p, self.fubini_study_metric_twist_V)
        d_correction = curvature.del_bar_z(p, self.exact_piece, False, params,
                                           H_fn_i, conf_params, frame_idx)
                                           # self.fubini_study_metric_twist_V, conf_params)
        d_correction = jnp.einsum("...abiu, ...ju->...abij", d_correction, jnp.conjugate(pb))
        return F_H0 + d_correction
    
    def trace_free_curvature_correction(self, p, pb, params, conf_params=None, frame_idx=None):
        F = self.curvature_correction(p, pb, params, conf_params, frame_idx)
        TrF = jnp.einsum("...aaij->...ij", F)
        trace_part = 1./self.rank_V * jnp.einsum("...ij, ...ab->...abij", 
            TrF, jnp.eye(self.rank_V, dtype=TrF.dtype))
        return F - trace_part


    def exact_piece(self, p, params, H0_metric_fn, conf_params=None, frame_idx=None):
        pb = self.pb_fn(math_utils.to_complex(p))
        # h = self.endomorphism_network(p, params, conf_params)  # h^b_a
        h = self.endomorphism_network(p, params, conf_params, True, frame_idx=frame_idx)    # h_a^b

        # dh = curvature.del_z(p, self.endomorphism_network, False, params, conf_params)  # h^b_{ai}
        dh = curvature.del_z(p, self.endomorphism_network, False, params, conf_params, True, frame_idx)  # h_{ai}^b
        dh = jnp.einsum("...abu, ...iu->...abi", dh, pb)
        A_0 = self.connection_form_V(p, H0_metric_fn)  # A_{ai}^b

        #_A1 = jnp.einsum("...aci, ...cb->...abi", A_0, h)
        #_A2 = jnp.einsum("...cbi, ...ac->...abi", A_0, h)

        _A1 = jnp.einsum("...ac, ...cbi->...abi", h, A_0)
        _A2 = jnp.einsum("...cb, ...aci->...abi", h, A_0)
        holo_cov_der_h = dh + _A1 - _A2
        exact = jnp.einsum("...aci, ...cb->...abi", holo_cov_der_h, jnp.linalg.inv(h))
        # exact = jnp.einsum("...ca, ...abi->...cbi", jnp.linalg.inv(h), holo_cov_der_h)
        #exact = jnp.linalg.solve(h, holo_cov_der_h.reshape(self.rank_V, -1))
        #exact = exact.reshape(self.rank_V, self.rank_V, self.cy_dim)
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


    def TrF_correction(self, p, pb, params, conf_params):
        F = self.curvature_correction(p, pb, params, conf_params)
        Tr_eta = jnp.einsum("...aaij->...ij", F)

        return Tr_eta

    def contract_TrF(self, p, pb, params):
        Tr_eta = self.TrF_correction(p, pb, params)
        g_inv = jnp.linalg.inv(self._metric_fn(p))  # \bar{\nu} \mu
        return jnp.einsum("...vu, ...uv->...", g_inv, Tr_eta)


    @partial(jax.jit, static_argnums=(0,))
    def codifferential_TrF(self, p, pb, params, conf_params=None):
        """
        Used to sanity-check harmonicity of Tr F.
        `params`: parameters for the endomorphism network
        """
        g_inv = jnp.linalg.inv(self._metric_fn(p))  # \bar{\nu} \mu
        TrF = self.TrF_correction(p, pb, params, conf_params)
        del_z_TrF = curvature.del_z(p, self.TrF_correction, False, pb, params, conf_params)  # [\mu, \bar{\nu}, \kappa]
        del_z_TrF = jnp.einsum("...iju, ...ku->...ijk", del_z_TrF, pb)

        Gamma_holo = curvature.christoffel_symbols_kahler(p, self._metric_fn, pb)  # [a, \kappa, b]
        _cov2 = jnp.einsum('...akb, ...av -> ...bvk', Gamma_holo, TrF)   # [b, \bar{\nu}, \kappa]
        covariant_derivative_eta = del_z_TrF - _cov2
        codiff = -jnp.einsum('...vu, ...bvu->...b', g_inv, covariant_derivative_eta)
        return codiff



    @partial(jax.jit, static_argnums=(0,))
    def objective_function(self, data, params, aux_params=None, frame_idx=None):
        (p, pb, w) = data
        #vol_Omega = jnp.mean(w)
        
        """
        loss = hym.objective_function_implicit_slope_V(data, params, 
                                                       self.curvature_correction,
                                                       self._metric_fn, self.section_metric_network, 
                                                       self.rank_V)
        return loss
        """
        MAX_YM_NORM = 500.
        g = vmap(self._metric_fn)(p)
        g_inv = jnp.linalg.inv(g)
        F = vmap(self.trace_free_curvature_correction, in_axes=(0,0,None,None,None))(p,
                pb, params, aux_params, frame_idx)
        Lambda_F0 = jnp.einsum("...vu, ...abuv->...ab", g_inv, F)
        #F = vmap(self.curvature_correction, in_axes=(0,0,None,None,None))(p,
        #        pb, params, aux_params, frame_idx)
        # H = vmap(self.fubini_study_metric_twist_V)(p)
        H = vmap(self.section_metric_network, in_axes=(0,None,None,None,None))(p, params, aux_params, True, frame_idx)
        Lambda_F0_norm = jnp.einsum("...ca, ...ab, ...cd, ...bd->...", jnp.linalg.inv(H), Lambda_F0, jnp.conj(Lambda_F0), H)
        energy = jnp.abs(Lambda_F0_norm) / 2.
        return jnp.mean(w * energy) / jnp.mean(w)

        #F_up = jnp.einsum("...ji, ...kl, ...abik->...abjl", g_inv, g_inv, F) #  F^{\bar{\nu} \mu}^a_b
        #F_sq = jnp.einsum("...abij, ...cdij, ...db, ...ac->...", F, jnp.conjugate(F_up), jnp.linalg.inv(H), H)
        # F_sq = jnp.einsum("...abij, ...cdkl, ...db, ...ac->...ijkl", F, jnp.conj(F), jnp.linalg.inv(H), H)
        F_sq = jnp.einsum("...abij, ...cdkl, ...ca, ...bd->...ijkl", F, jnp.conj(F), jnp.linalg.inv(H), H)
        F_sq = jnp.einsum("...ijkl, ...ki, ...jl->...", F_sq, g_inv, g_inv)

        ym_energy = jnp.abs(F_sq) / 2.

        valid = jnp.logical_and(jnp.isfinite(ym_energy), ym_energy < MAX_YM_NORM)
        valid_w = w * valid.astype(w.dtype)
        #ym_energy = jnp.where(ym_energy < MAX_YM_NORM, ym_energy, 0.)
        #return ym_energy
        return jnp.mean(valid_w * ym_energy) / jnp.mean(valid_w)
        return jnp.mean(w * ym_energy) / jnp.mean(w)
        return loss

    def conformal_rescale_network(self, p, params):
        if (self.n_hyper > 1) or (len(self.ambient) > 1):
            model_class = models.LearnedVector_spectral_nn_CICY
        else:
            model_class = models.LearnedVector_spectral_nn
        f = model_class(p.shape[-1]//2, tuple(self.ambient), self.n_units,
                        n_out=1, activation=self.activation).apply({'params': params}, p)
        return f

    def conformal_change(self, p, params, endo_params=None):
        pb = self.pb_fn(math_utils.to_complex(p))
        if endo_params is None:
            xi = self.ddbar_log_det_H_0(p, pb)
        else:
            xi = self.ddbar_log_det_H(p, pb, endo_params, conformal_factor=False)
        # xi = self.TrF_H_0(p)
        # xi = curvature.ricci_form_kahler(p, self.fs_metric_fn, pb)
        ddbar_f = self.del_z_del_z_bar(p, self.conformal_rescale_network, params)
        ddbar_f = jnp.einsum("...iu, ...uv, ...jv->...ij", pb, ddbar_f, jnp.conjugate(pb))
        return xi + self.rank_V * ddbar_f

    @partial(jax.jit, static_argnums=(0,))
    def codifferential_TrF_conformal(self, p, pb, params, endo_params=None):

        g_inv = jnp.linalg.inv(self._metric_fn(p))  # \bar{\nu} \mu
        TrF = self.conformal_change(p, params, endo_params)
        del_z_TrF = curvature.del_z(p, self.conformal_change, False, params, endo_params)  # [\mu, \bar{\nu}, \kappa]
        del_z_TrF = jnp.einsum("...iju, ...ku->...ijk", del_z_TrF, pb)

        Gamma_holo = curvature.christoffel_symbols_kahler(p, self._metric_fn, pb)  # [a, \kappa, b]
        _cov2 = jnp.einsum('...akb, ...av -> ...bvk', Gamma_holo, TrF)   # [b, \bar{\nu}, \kappa]
        covariant_derivative_eta = del_z_TrF - _cov2
        codiff = jnp.einsum('...vu, ...bvu->...b', g_inv, covariant_derivative_eta)
        return -codiff

    @partial(jax.jit, static_argnums=(0,))
    def objective_function_conformal(self, data, params, endo_params=None, 
                                     MAX_NORM=20.):
        (p, pb, w) = data
        vol_Omega = jnp.mean(w)
        codiff = vmap(self.codifferential_TrF_conformal, in_axes=(0,0,None,None))(p, pb, 
            params, endo_params)
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
        F_H_0 = self.curvature_form_V(p, self.fubini_study_metric_twist_V)
        return jnp.einsum("...aaij->...ij", F_H_0)
    
    def ddbar_log_det_H(self, p, pb, endo_params, conf_params=None, conformal_factor=False):
        hess = self.del_z_del_z_bar(p, self.log_det_H, endo_params, conf_params, conformal_factor)
        return jnp.einsum("...iu, ...uv, ...jv->...ij", pb, hess, jnp.conjugate(pb))

    def log_det_H(self, p, endo_params, conf_params, conformal_factor=False):
        H = self.section_metric_network(p, endo_params, conf_params, conformal_factor=conformal_factor)
        s, logdet = jnp.linalg.slogdet(H)
        return logdet + 1j * jnp.pi * (s < 0)

    def ddbar_log_det_H_0(self, p, pb):
        hess = self.del_z_del_z_bar(p, self.log_det_H_0)
        return jnp.einsum("...iu, ...uv, ...jv->...ij", pb, hess, jnp.conjugate(pb))

    def log_det_H_0(self, p):
        H_0_inv = self.fubini_study_metric_twist_V(p, True)
        s, logdet = jnp.linalg.slogdet(H_0_inv)
        return -logdet + 1j * jnp.pi * (s < 0)

    def H0_conformal_change(self, p, params):
        f = self.conformal_rescale_network(p, params)
        H0 = self.fubini_study_metric_twist_V(p)
        return jnp.expand_dims(jnp.exp(f), (0,1)) * H0

    def embedding_matrix_twisted(self, p, patch_idx=None):
        r"""Describes relationship between ambient frame vectors induced
            by the monad map.
        """
        if patch_idx is None:
            patch_idx = jnp.argmax(jnp.abs(math_utils.to_complex(p))[:self.rank_B])
        proj = jnp.eye(self.rank_V, dtype=self.cdtype)
        f_p = poly_utils.monomial_evaluate_log(p, self.monad_map_power_matrix)
        col = -f_p / f_p[patch_idx]  # f_p[patch_idx] should usually be 1.
        col = jnp.delete(col, patch_idx, assume_unique_indices=True)

        return jnp.insert(proj, patch_idx, col, axis=-1)

    def _trivial_mono_index_Pn(self, Ok_powers, coord_j, k):
        # Find idx of Z_j^k in the monomial power matrix
        if self.line_bundle_A_twist > 0:
            return self._trivial_mono_index_mult_Pn(Ok_powers, coord_j, k)
        target = jnp.zeros((Ok_powers.shape[1],), Ok_powers.dtype).at[coord_j].set(k)
        mask = jnp.all(Ok_powers == target, axis=1)
        # assumes target exists; fall back to 0 if not found
        return jnp.where(mask, size=1, fill_value=0)[0][0]

    def _trivial_mono_index_mult_Pn(self, Ok_powers, coord_j, k_twist):
        n_coords = Ok_powers.shape[1]
        target = jnp.zeros((n_coords,), Ok_powers.dtype).at[coord_j].set(1)
        mask = jnp.all((Ok_powers - target) >= 0, axis=1)
        # return jnp.where(mask, fill_value=0)[0]
        return jnp.where(mask, size=self.n_quotient, fill_value=0)[0]

    def twisted_section_basis(self, p, frame_idx=None, ambient=False):

        p_c = math_utils.to_complex(p)
        if frame_idx is None:
            frame_idx = jnp.argmax(jnp.abs(p_c)[:self.rank_B])

        Ok_powers = self.degree_to_monomial_basis[self.twisting_degree]
        Ok_monomials = poly_utils.monomial_evaluate_log(p, Ok_powers)
        blocks = [Ok_monomials] * self.rank_B
        section_matrix = jax.scipy.linalg.block_diag(*blocks)  # frame-independent
        n_Ok = Ok_monomials.shape[0]

        # choose ambient coord used to trivialise O(k); for P^n pick the dominant homo coord
        trivial_idx = self._trivial_mono_index_Pn(Ok_powers, coord_j=frame_idx, k=self.twisting_degree)
        col_to_delete = frame_idx * n_Ok + trivial_idx
        # col_to_delete = 0
        # print(col_to_delete)
        section_matrix = jnp.delete(section_matrix, col_to_delete, axis=-1, assume_unique_indices=True)

        if ambient is True:
            return section_matrix
        emb = self.embedding_matrix_twisted(p, frame_idx)
        return emb @ section_matrix


    def twisted_section_basis_in_frame(self, p, frame_idx=None, drop_patch_idx=None,
            ambient: bool = False):
        """
        Sections of V ⊗ O(k) expressed in a fixed fiber frame 'frame_idx',
        dropping exactly one column: the trivial monomial Z_{frame_idx}^k
        from the block 'drop_patch_idx'. This lets you compare bases without
        applying a row transition T.
        """

        p_c = math_utils.to_complex(p)
        if frame_idx is None:
            frame_idx = jnp.argmax(jnp.abs(p_c)[:self.rank_B])
        if drop_patch_idx is None:
            drop_patch_idx = frame_idx

        Ok_powers = self.degree_to_monomial_basis[self.twisting_degree]
        Ok_monomials = poly_utils.monomial_evaluate_log(p, Ok_powers)
        blocks = [Ok_monomials] * self.rank_B
        section_matrix = jax.scipy.linalg.block_diag(*blocks)  # frame-independent
        n_Ok = Ok_monomials.shape[0]

        # trivial monomial position relative to the chosen frame
        trivial_idx = self._trivial_mono_index_Pn(Ok_powers, coord_j=drop_patch_idx, k=self.twisting_degree)
        col_to_delete = drop_patch_idx * n_Ok + trivial_idx
        # print('tsb: deleting', col_to_delete)
        section_matrix = jnp.delete(section_matrix, col_to_delete, axis=-1, assume_unique_indices=True)

        if ambient is True:
            return section_matrix
        emb = self.embedding_matrix_twisted(p, frame_idx)  # rows in 'frame_idx'
        return emb @ section_matrix


    def H0XV_transition_matrix(self, p, new_frame_idx=0):
        """
        `old_frame_idx`: Basis vector to be added when transitioning between patches. 
                         This is `i` for a transition `P_{i->j}`. This is the dynamic frame.
        `new_frame_idx`: Basis vector to be removed when transitioning between patches. 
                         This is `j` for a transition `P_{i->j}`. This is the default frame.
        """
        old_frame_idx = jnp.argmax(jnp.abs(math_utils.to_complex(p))[:self.rank_B])
        P = jnp.eye(self._N_sb, dtype=self.cdtype)
        Ok_powers = self.degree_to_monomial_basis[self.twisting_degree]
        n_Ok = len(Ok_powers)

        old_trivial_idx = self._trivial_mono_index_Pn(Ok_powers, old_frame_idx, self.twisting_degree)
        new_trivial_idx = self._trivial_mono_index_Pn(Ok_powers, new_frame_idx, self.twisting_degree)

        scatter = vmap(self._trivial_mono_index_Pn, in_axes=(None,0,None))(Ok_powers, jnp.arange(self.rank_B), self.twisting_degree)
        elim_cols_idx = jnp.arange(self.rank_B) * n_Ok + scatter
        elim_cols_idx = jnp.delete(elim_cols_idx, old_frame_idx, assume_unique_indices=True)
        #print('info added at rows',elim_cols_idx)

        # choose ambient coord used to trivialise O(k); for P^n pick the dominant homo coord
        old_elim_col = old_frame_idx * n_Ok + old_trivial_idx  # free to choose this
        new_elim_col = new_frame_idx * n_Ok + new_trivial_idx  # this is frame-independent

        col = jnp.zeros(self._N_sb, dtype=self.cdtype)

        for i, elim_idx in enumerate(elim_cols_idx):
            shift = old_elim_col < elim_idx
            col = col.at[elim_idx-shift].set(-1)
        
        #print('inserting', old_elim_col)
        P = jnp.insert(P, old_elim_col, col, axis=1)
        #print('deleting', new_elim_col)
        P = jnp.delete(P, new_elim_col, axis=1, assume_unique_indices=True)
        return P

    def fubini_study_metric_twist_V(self, p, inv=False, frame_idx=None, transport=False):
        if frame_idx is None:
            frame_idx = jnp.argmax(jnp.abs(math_utils.to_complex(p))[:self.rank_B])
        #tsb = self.twisted_section_basis(p, frame_idx=frame_idx)
        tsb = self.twisted_section_basis_in_frame(p, frame_idx=frame_idx, drop_patch_idx=self.default_idx)

        if transport is True: 
            P = self.H0XV_transition_matrix(p, self.default_idx)
            _H = P @ self.dagger(P)
            fs_inv = jnp.einsum("...am, ...mn, ...bn->...ba", tsb, _H, jnp.conjugate(tsb))
        else:
            fs_inv = jnp.einsum("...am, ...bm->...ba", tsb, jnp.conjugate(tsb))
        if inv is True: return fs_inv
        fs = jnp.linalg.inv(fs_inv)
        return fs

    def _fubini_study_metric_twist_V(self, p, inv=False, frame_idx=None):
        tsb = self.twisted_section_basis(p, frame_idx=frame_idx)
        fs_inv = jnp.einsum("...am, ...bm->...ba", tsb, jnp.conjugate(tsb))
        if inv is True: return fs_inv
        fs = jnp.linalg.inv(fs_inv)
        return fs

    def transition_function(self, p, old_frame_idx, new_frame_idx):
        """
        Move from local trivialisation `old_frame_idx` to 
        `new_frame_idx`
        """
        T = jnp.eye(self.rank_V, dtype=self.cdtype)
        f_p = poly_utils.monomial_evaluate_log(p, self.monad_map_power_matrix)
        col = -f_p / f_p[new_frame_idx]  # f_p[patch_idx] should usually be 1.
        col = jnp.delete(col, new_frame_idx, assume_unique_indices=True)
        T = jnp.insert(T, new_frame_idx, col, axis=-1)
        T = jnp.delete(T, old_frame_idx, assume_unique_indices=True, axis=1)
        return T


    def section_metric_network(self, p, params, conf_params=None, conformal_factor=True, frame_idx=None):
        r"""
        Returns a smooth section of $Sym(V^* \otimes V^*$) from basis of sections for $V$.
        coeffs = models.cholesky_head(p, params, self.n_homo_coords, tuple(self.ambient), self._N_sb)
        tsb = self.twisted_section_basis(p)
        G_inv = jnp.einsum("...am, ...mn, ...bn->...ab", jnp.conj(tsb), coeffs, tsb)
        return jnp.linalg.inv(G_inv)
        """
        h = self.endomorphism_network(p, params, conf_params, conformal_factor, frame_idx)
        H0 = self.fubini_study_metric_twist_V(p, frame_idx=frame_idx)
        # H = jnp.einsum("...ca, ...cb->...ab", h, H0)
        H = jnp.einsum("...ac, ...cb->...ab", h, H0)
        return H

    @staticmethod
    def low_rank_reconstruct(M, D, S, S_dual):
        U = S @ M
        V = jnp.conj(M).T @ S_dual.T
        h_lr = U @ V
        h_diag = jnp.einsum("...am, ...m, ...bm->...ab", S, D, S_dual)
        h = h_lr + h_diag
        return h

    def endomorphism_network(self, p, params, conf_params=None, conformal_factor=True, frame_idx=None):
        r"""
        Model a section of the endomorphism bundle on $V$ as a matrix of coefficients (each of which is 
        a global function), which parameterise the section via a linear combination of a section of 
        $V$ tensored by a dual section.
        $$ h^b_a = \sum_{mn} H^{mn} S^b_m \otimes \hat{S}_{an}~. $$
        """
        h0 = jnp.eye(self.rank_V, dtype=self.cdtype)
        # TODO
        coeffs = models.cholesky_head(p, params, self.n_homo_coords, tuple(self.ambient),
                                      self._N_sb, normalise_det=False, n_frames=self.n_frames,
                                      low_rank_approx=self.lr_approx)

        # CHANGES
        if frame_idx is None:
            frame_idx = jnp.argmax(jnp.abs(math_utils.to_complex(p))[:self.rank_B])
        # P = self.H0XV_transition_matrix(p, self.default_idx)
        # coeffs = P @ coeffs @ self.dagger(P)
        # coeffs = jnp.einsum("...xm, ...mn, ...yn->...xy", P, coeffs, jnp.conj(P))
        # CHANGES
        #tsb = self.twisted_section_basis(p)
        tsb = self.twisted_section_basis_in_frame(p, frame_idx=frame_idx, drop_patch_idx=self.default_idx)
        H_fs_V = self.fubini_study_metric_twist_V(p, frame_idx=frame_idx, transport=False)
        tsb_dual = jnp.einsum("...ab, ...bm->...am", H_fs_V, jnp.conjugate(tsb))

        if self.lr_approx > 0:
            M, D = coeffs
            h = self.low_rank_reconstruct(M, D, tsb, tsb_dual)
        else:  # full dense matrix
            #h = jnp.einsum("...am, ...mn, ...bn->...ab", tsb, coeffs, tsb_dual)
            h = jnp.einsum("...am, ...mn, ...bn->...ab", tsb_dual, coeffs, tsb)

        # return h
        log_h_raw = h
        # Project to sl(n,C)
        tr = jnp.trace(log_h_raw)
        log_h = log_h_raw - tr / self.rank_V * jnp.eye(self.rank_V, dtype=self.cdtype)

        # Matrix exponential -> unit determinant endomorphism
        h = jax.scipy.linalg.expm(log_h)

        if conformal_factor is True and conf_params is not None:
            f = self.conformal_rescale_network(p, conf_params)
            return h * jnp.exp(f)
        return h

    @partial(jax.jit, static_argnums=(0,))
    def untwisted_metric(self, p, params, conf_params=None):
        # Untwist metric on $V \otimes \mathcal{L}^k$ with determinant bundle
        H_K = self.section_metric_network(p, params, conf_params)
        det_H_K = jnp.linalg.det(H_K)
        return H_K * (det_H_K)**(-1./self.rank_V)

    @partial(jax.jit, static_argnums=(0,))
    def untwisted_curvature(self, p, params, conf_params=None):
        if conf_params is not None:
            untwisted_metric = jax.tree_util.Partial(self.untwisted_metric, 
                    conf_params=conf_params)
        else:
            untwisted_metric = self.untwisted_metric

        return self.curvature_form_V(p, untwisted_metric, params)

    @staticmethod
    def int_dVol_Omega(f, w, vol_w):
        return jnp.mean(f * w) / vol_w

    @partial(jax.jit, static_argnums=(0,))
    def loss_breakdown(self, data, params, conf_params=None):
        
        loss = self.objective_function(data, params, conf_params)
        p, pbs, w = data
        vol_Omega = jnp.mean(w)

        h = vmap(self.endomorphism_network, in_axes=(0,None,None))(p, params, conf_params)
        g = vmap(self._metric_fn)(p)
        g_inv = jnp.linalg.inv(g)
        F = vmap(self.trace_free_curvature_correction, in_axes=(0,0,None,None))(p,
                pbs, params, conf_params)
        H = vmap(self.section_metric_network, in_axes=(0,None,None))(p, params, conf_params)

        k = p.shape[0]//2
        codiff_TrF = vmap(self.codifferential_TrF, in_axes=(0,0,None,None))(p, pbs, params, conf_params)
        abs_codiff = jnp.mean(jnp.abs(codiff_TrF), axis=-1)        
        codiff_mean = jnp.mean(abs_codiff * w) / jnp.mean(w)

        g_tr_F = jnp.einsum("...vu, ...abuv->...ab", g_inv, F)
        Lambda_F0 = g_tr_F
        Lambda_F_sq = jnp.einsum("...ab, ...bc->...ac", g_tr_F, g_tr_F)
        Tr_Lambda_F_sq = jnp.einsum("...aa->...", Lambda_F_sq)
        F_sq = jnp.einsum("...abuv, ...bcuv->...acuv", F, F)
        g_tr_F_sq = jnp.einsum("...vu, ...abuv->...ab", g_inv, F_sq)
        det_g_tr_F = jnp.linalg.det(g_tr_F)
        max_eig = vmap(jnp.linalg.norm)(g_tr_F)
        Tr_F_g = vmap(jnp.trace)(g_tr_F)
        Tr_F_sq_g = vmap(jnp.trace)(g_tr_F_sq)

        Lambda_F0_norm = jnp.einsum("...ca, ...ab, ...cd, ...bd->...", jnp.linalg.inv(H), Lambda_F0, jnp.conj(Lambda_F0), H)
        energy = jnp.abs(Lambda_F0_norm) / 2.

        H_ut = vmap(self.untwisted_metric, in_axes=(0,None,None))(p, params, conf_params)
        H_ut_integrand = jnp.expand_dims(w, (1,2)) * H_ut
        var_H_ut = jnp.mean(jnp.expand_dims(w, (1,2)) * H_ut**2) / vol_Omega - jnp.mean(H_ut_integrand)**2 / vol_Omega**2
        var_H = jnp.mean(jnp.expand_dims(w, (1,2)) * H**2) / vol_Omega - jnp.mean(jnp.expand_dims(w, (1,2)) * H)**2 / vol_Omega**2
        tr_H_ut = vmap(jnp.trace)(H_ut)
        upper_bound_var = hym.objective_function_implicit_slope_V(data, params,
                                                       self.trace_free_curvature_correction,
                                                       self._metric_fn, self.section_metric_network,
                                                       self.rank_V, conf_params)

        report = {'loss': loss, 'Tr_F_g': jnp.mean(w * Tr_F_g) / vol_Omega, "max_eig": jnp.mean(w * max_eig) / vol_Omega,
                'det_F_g': jnp.mean(w * det_g_tr_F) / vol_Omega, "det_h": jnp.mean(w * jnp.linalg.det(h)) / vol_Omega,
                'Tr_F_g_var': jnp.var(jnp.abs(g_tr_F)), 'Tr_Lambda_F_sq': jnp.mean(w * Tr_Lambda_F_sq) / vol_Omega,
                'Λ F_0 energy': jnp.mean(w * energy) / vol_Omega, 'codiff_mean': codiff_mean,
                'H': jnp.mean(jnp.expand_dims(w,(1,2)) * H, axis=0) / vol_Omega,
                'H_untwist': jnp.mean(jnp.expand_dims(w,(1,2)) * H_ut, axis=0) / vol_Omega, 'upper_bound_var': upper_bound_var, 
                'var_H_ut': var_H_ut, 'var_H': var_H}

        if conf_params is not None:
            f = vmap(self.conformal_rescale_network, in_axes=(0,None))(p, conf_params)
            conf_loss = self.objective_function_conformal(data, conf_params, params)
            codiff_TrF_conf = vmap(self.codifferential_TrF_conformal, in_axes=(0,0,None))(p, pbs, conf_params)
            abs_codiff_conf = jnp.mean(jnp.abs(codiff_TrF_conf), axis=-1)        
            codiff_mean_conf = jnp.mean(abs_codiff_conf * w) / vol_Omega
            TrF = vmap(self.conformal_change, in_axes=(0,None))(p, conf_params)
            Lambda_TrF = jnp.einsum("...vu, ...uv->... ", g_inv, TrF)
            report.update({'conformal_loss': conf_loss, "Λ Tr F": jnp.mean(w * Lambda_TrF) / vol_Omega,
                           'codiff_conf': codiff_mean_conf, "f_avg": jnp.mean(w * f) / vol_Omega})

        return report

    def loss_breakdown_conformal(self, data, params):

        loss = self.objective_function_conformal(data, params)
        p, pb, w = data
        f = vmap(self.conformal_rescale_network, in_axes=(0,None))(p, params)
        H0 = vmap(self.fubini_study_metric_twist_V)(p)
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
                 slope: float = None, conformal_train=False, conf_params=None):
        
        if conformal_train is True:
            loss_breakdown_dict = self.loss_breakdown_conformal(val_data, params)
        else:
            loss_breakdown_dict = self.loss_breakdown(val_data, params, conf_params)

        loss_breakdown_dict = jax.device_get(loss_breakdown_dict)
        # summary = jax.tree_util.tree_map(lambda x: x.item(), loss_breakdown_dict)
        summary = jax.tree_util.tree_map(utils.log_arrays, loss_breakdown_dict)


        mode = 'VAL'
        # logs = [f"{k}: {v:.4f}" for (k,v) in summary.items()]        
        logs = [utils.round_str(k, v) for (k, v) in summary.items()]
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
            conformal_fn = None, shuffle_rng = np.random.default_rng(), name = None,
            ckpt: dict = None):

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

        # _tx = optax.adamw(learning_rate=lr)
        grad_threshold = 1.
        _tx = optax.chain(
          optax.clip(grad_threshold),
          optax.adamw(learning_rate=lr),
        )
        self.n_units_harmonic = [48,48,48]
        if conformal_fn is not None: self.conformal_fn = conformal_fn

        coeff_class = models.CholeskyNetwork
        bundle_metric_model = coeff_class(self.n_homo_coords, self.ambient, self.n_units_harmonic,
                matrix_dim=self._N_sb, n_frames=self.n_frames, low_rank_approx=self.lr_approx)

        _params, _opt_state, _ = create_train_state(_k, bundle_metric_model, _tx, data_dim=self.n_homo_coords * 2)
        # _params, _opt_state, _ = self._create_train_state(_k, bundle_metric_model, _tx)
        if ckpt is not None:
            _params, _opt_state = utils.load_ckpt(_params, _opt_state, ckpt['params'], ckpt['opt'])
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(_params))
        logger.info(f'Params (Count: {param_count})=========>>>')
        logger.info(jax.tree_util.tree_map(lambda x: x.shape, _params))
        logger.info(bundle_metric_model.tabulate(_k, jnp.ones([1, self.n_homo_coords * 2])))
        # logger.info(bundle_metric_model.tabulate(_k)) 

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


    def fit_alternating(self,
                        data_path,
                        epochs: int = 32,
                        batch_size: int = 512,
                        lr_conf: float = 1e-4,
                        lr_endo: float = 1e-4,
                        inner_conf: int = 16,
                        inner_endo: int = 1,
                        shuffle_rng = np.random.default_rng(),
                        name: str = None,
                        ckpt_conf: dict = None,
                        ckpt_endo: dict = None):

        self.name = f"HYM_alt_{datetime.now().strftime('%Y-%m-%d_%H')}" if name is None else name
        self.eval_interval = 1
        self.save_interval = 8
        self.eval_interval_t = 512

        storage = defaultdict(list)
        logger = utils.logger_setup(self.name, filepath=os.path.abspath(__file__))
        data_path = os.path.join(data_path, 'dataset.npz')
        os.makedirs(os.path.join("experiments", self.name), exist_ok=True)
        logger.info(f'Dataset: {data_path}')

        A_train, A_val, train_loader, val_loader, psi = dataloading.initialize_loaders_train(
            shuffle_rng, data_path, batch_size, logger=logger)
        dataset_size = A_train[0].shape[0]

        # slope normalization
        vol = jnp.mean(A_train[1])
        if self._slope is not None:
            self._slope *= vol

        try:
            device = jax.devices('gpu')[0]
        except Exception:
            print("gpu not detected, falling back to cpu.")
            device = jax.devices('cpu')[0]

        # models
        key = jax.random.key(42)
        key, k_conf, k_endo = jax.random.split(key, 3)

        # conformal network
        self.n_units = [48, 48, 48]
        conf_model = models.LearnedVector_spectral_nn(self.n_homo_coords, self.ambient, self.n_units)
        tx_conf = optax.chain(optax.clip(1.0), optax.adamw(learning_rate=lr_conf))
        conf_params, conf_opt_state, _ = create_train_state(k_conf, conf_model, tx_conf, data_dim=self.n_homo_coords * 2)

        # endomorphism (coeff) network
        self.n_units_harmonic = [48, 48, 48]
        coeff_model = models.CholeskyNetwork(self.n_homo_coords, self.ambient, self.n_units_harmonic,
                                             matrix_dim=self._N_sb, n_frames=self.n_frames,
                                             low_rank_approx=self.lr_approx)
        tx_endo = optax.chain(optax.clip(1.0), optax.adamw(learning_rate=lr_endo))
        endo_params, endo_opt_state, _ = create_train_state(k_endo, coeff_model, tx_endo, data_dim=self.n_homo_coords * 2)
        # endo_params, endo_opt_state, _ = self._create_train_state(k_endo, coeff_model, tx_endo)

        # optional restore
        if ckpt_conf is not None:
            conf_params, conf_opt_state = utils.load_ckpt(conf_params, conf_opt_state, ckpt_conf['params'], ckpt_conf['opt'])
        if ckpt_endo is not None:
            endo_params, endo_opt_state = utils.load_ckpt(endo_params, endo_opt_state, ckpt_endo['params'], ckpt_endo['opt'])

        # logs
        logger.info("Conformal model params: %d", sum(x.size for x in jax.tree_util.tree_leaves(conf_params)))
        logger.info(conf_model.tabulate(k_conf, jnp.ones([1, self.n_homo_coords * 2])))
        logger.info("Endomorphism model params: %d", sum(x.size for x in jax.tree_util.tree_leaves(endo_params)))
        logger.info(coeff_model.tabulate(k_endo, jnp.ones([1, self.n_homo_coords * 2])))
        # logger.info(coeff_model.tabulate(k_endo))

        # training
        t0 = time.time()
        iter_loader = iter(train_loader)
        with jax.default_device(device):

            for epoch in range(epochs):

                # validation
                if epoch % self.eval_interval == 0:
                    val_loader, val_data = dataloading.get_validation_data(val_loader, batch_size, A_val, shuffle_rng)
                    p, w, _ = val_data
                    pb = vmap(self.pb_fn)(math_utils.to_complex(p))
                    val_data = (p, pb, w)

                    storage = self.callback(val_data, endo_params, storage, logger, epoch, t0, self._slope,
                                            conf_params=conf_params)

                # get fresh train loader after first epoch
                if epoch > 0:
                    train_loader = dataloading.data_loader(A_train, batch_size, shuffle_rng)
                    iter_loader = iter(train_loader)

                # alternating blocks
                pbar = tqdm(desc=f"Epoch {epoch} [alt]",
                            total=dataset_size // batch_size,
                            dynamic_ncols=True, leave=False, colour='green',
                            mininterval=0.1)

                last_conf = None
                last_endo = None

                def _postfix():
                    c = f"{last_conf:.5f}" if last_conf is not None else "—"
                    e = f"{last_endo:.5f}" if last_endo is not None else "—"
                    return f"conf: {c} | endo: {e}"


                # for t, batch in enumerate(train_loader):
                t = 0
                while True:
                    try:
                        # 1) K_conf conformal updates
                        for _ in range(inner_conf):
                            p, w, _ = next(iter_loader)
                            pb = vmap(self.pb_fn)(math_utils.to_complex(p))
                            data = (p, pb, w)
                            conf_params, conf_opt_state, loss_conf = hym._train_step(
                                data, conf_params, conf_opt_state, tx_conf, self.objective_function_conformal,
                                aux_params=endo_params
                            )
                            last_conf = loss_conf.item()
                            pbar.set_postfix_str(_postfix(), refresh=False)
                            pbar.update(1)

                        # 2) K_endo endomorphism updates
                        for _ in range(inner_endo):
                            p, w, _ = next(iter_loader)
                            pb = vmap(self.pb_fn)(math_utils.to_complex(p))
                            data = (p, pb, w)
                            endo_params, endo_opt_state, loss_endo = hym._train_step(
                                data, endo_params, endo_opt_state, tx_endo, self.objective_function,
                                aux_params=conf_params
                            )
                            last_endo = loss_endo.item()
                            pbar.set_postfix_str(_postfix(), refresh=False)
                            pbar.update(1)

                        t += 1
                        if t % self.eval_interval_t == 0:
                            storage["train_loss_conf"].append(loss_conf.item())
                            storage["train_loss_endo"].append(loss_endo.item())

                            val_loader, val_data = dataloading.get_validation_data(val_loader, batch_size, A_val, shuffle_rng)
                            p, w, _ = val_data
                            pb = vmap(self.pb_fn)(math_utils.to_complex(p))
                            val_data = (p, pb, w)

                            storage = self.callback(val_data, endo_params, storage, logger, epoch, t0, self._slope,
                                                    conf_params=conf_params)
                    except StopIteration:
                        break

                    # pbar.update(1)
                pbar.close()

                if epoch % self.save_interval == 0:
                    utils.basic_ckpt({'conf': conf_params, 'endo': endo_params},
                                     {'conf': conf_opt_state, 'endo': endo_opt_state},
                                     self.name, f'{epoch}')

        utils.basic_ckpt({'conf': conf_params, 'endo': endo_params},
                         {'conf': conf_opt_state, 'endo': endo_opt_state},
                         self.name, 'FIN')
        utils.save_logs(storage, self.name, 'FIN')
        return {'conf': conf_params, 'endo': endo_params}, storage


class GenDonaldson(HarmonicBundle):

    def __init__(self, metric_fn, monomials, coefficients, cy_dim, ambient, defining_polys=None):
        super().__init__(metric_fn, monomials, coefficients, cy_dim, ambient, defining_polys)

        self.rank_V = 3
        self.twisting_degree = 1
        self.line_bundle_B = (1,1,1,1)
        self.rank_B = len(self.line_bundle_B)
        self.monad_map_power_matrix = self.monad_map_power_matrix_DKLR
        self._N_sb = len(self.degree_to_monomial_basis[self.twisting_degree]) * self.rank_B - 1

    @staticmethod
    def project_Hermitian(M):
        return (M + jnp.conjugate(M).T) * 0.5
    
    def untwisted_metric(self, p, H):
        # Untwist metric on $V \otimes \mathcal{L}^k$ with determinant bundle
        H_K = self.fibre_metric_from_H(p, H)
        det_H_K = jnp.linalg.det(H_K)
        return H_K * (det_H_K)**(-1./self.rank_V)
        
    def untwisted_curvature(self, p, H):
        return self.curvature_form_V(p, self.untwisted_metric, H)

    def sample_intersect_hypersurface(self, key: random.PRNGKey, n_p: int, 
                                      LOCUS_TOL: float = 1e-10):
        
        _key, key = random.split(key, 2)
        c_dim = self.cy_dim + 2  # homo. coords plus hypersurface constraint
        n_intersect = np.ceil(n_p / c_dim).astype(int)
        sphere_pts = pointgen.S2np1_uniform(_key, 2*n_intersect, self.cy_dim+1)
        p, q = jnp.split(sphere_pts, 2)

        t_coeffs_data, generators = pointgen.univariate_coefficient_data(self.cy_dim, self.monomials, self.coefficients)
        
        pts, t_coeffs = vmap(pointgen.root_solver, in_axes=(0,0,None,None,None))(
            p, q, t_coeffs_data.values(), tuple(generators), jax.devices()[0])
        pts = pts.reshape(-1, c_dim)

        pts, *_ = math_utils.rescale(pts.reshape(-1, c_dim)[:n_p])
        return pts

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
    

    def fibre_metric_from_H(self, p, H, aux=False):
        # S = self.twisted_section_basis(p)
        S = self.twisted_section_basis_in_frame(p, None, self.default_idx)
        S_c = jnp.conjugate(S)
        G_inv = jnp.einsum("mn, ...am, ...bn->...ab", H, S, S_c)
        G = jnp.einsum("...ba->...ab", jnp.linalg.inv(G_inv))  # G_{a \overline{b}}
        if aux is True: return G, S, S_c
        return G
            
    def fibre_metric_from_H_basis_change(self, p, H, aux=False):
        S = self.twisted_section_basis(p)
        S_c = jnp.conjugate(S)
        frame_idx = jnp.argmax(jnp.abs(math_utils.to_complex(p))[:self.rank_B])
        P = self.H0XV_transition_matrix(p, frame_idx, 0)
        P_inv = jnp.linalg.solve(P, jnp.eye(P.shape[-1]))  # jnp.linalg.inv(P)
        _H = P_inv @ H @ self.dagger(P_inv)
        G_inv = jnp.einsum("...mn, ...am, ...bn->...ab", _H, S, S_c)
        G = jnp.einsum("...ba->...ab", jnp.linalg.inv(G_inv))  # G_{a \overline{b}}
        if aux is True: return G, S, S_c
        return G

    @partial(jax.jit, static_argnums=(0,))
    def _G_update(self, p, H, w):
        p = math_utils.to_real(p)
        G, S, S_c = vmap(self.fibre_metric_from_H, in_axes=(0,None,None))(p, H, True)
        # G, S, S_c = vmap(self.fibre_metric_from_H_basis_change, in_axes=(0,None,None))(p, H, True)
        integrand = jnp.einsum("...am, ...ab, ...bn->...mn", S, G, S_c)
        
        delta_T = jnp.mean(integrand * jnp.expand_dims(w, axis=(1,2)), axis=0)
        vol_Omega_batch = jnp.mean(w)
        return delta_T, vol_Omega_batch


    @staticmethod
    def check_min(p, patch, threshold=1e-1):
        mask = jnp.abs(p)[:,patch] > threshold
        return math_utils.rescale_patch(p[mask], patch)

    @staticmethod
    def cutoff(p, threshold=1e-2):
        mask = jnp.min(jnp.abs(p), axis=-1) > threshold
        return p[mask]

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
            # p = self.cutoff(p)
            p = self.check_min(p, patch=0)  # move to patch 0
            B = p.shape[0]
            w, *_ = vmap(alg_geo.compute_integration_weights, in_axes=(0,None,None,None))(
                p, self.dQdz_monomials, self.dQdz_coeffs, self.cy_dim)

            delta_T, vol_Omega_batch = self._G_update(p, H, w)

            T = math_utils.online_update_array(T, delta_T, n, B)
            vol_Omega = math_utils.online_update(vol_Omega, vol_Omega_batch, n, B)
            n += B

        T = T * H.shape[0] / vol_Omega / self.rank_V

        # inverse of output of T-operator is the Hermitian matrix on the space of sections
        H_update = jnp.linalg.inv(T).T
        return H_update, math_utils.to_real(p), w

    def eval(self, p, w, H):

        vol_Omega = jnp.mean(w)
        g = vmap(self._metric_fn)(p)
        g_inv = jnp.linalg.inv(g)
        F = vmap(self.curvature_form_donaldson, in_axes=(0,None))(p, H)
        g_tr_F = jnp.einsum("...ji,...abij->...ab", g_inv, F)

        Lambda_TrF_pw = jnp.einsum("...aa->...", g_tr_F)
        Lambda_TrF = jnp.mean(Lambda_TrF_pw * w) / vol_Omega
        S_Lambda_TrF = jnp.sum((jnp.real(Lambda_TrF_pw) - jnp.real(Lambda_TrF))**2 * w) / vol_Omega
        sigma = jnp.sqrt(S_Lambda_TrF / (p.shape[0] - 1))
        return Lambda_TrF, sigma

    def generalised_donaldson(self, iterations=16, batch_size=8192):

        key = jax.random.PRNGKey(42)
        Np = (10 * self._N_sb**2 + 50000) * 10
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
            H, _p, _w = step(key, H, i)
            # H = H / ((det_H + eps)**(1.0 / H.shape[0]))
            H = H / jnp.max(jnp.abs(H))  # fix scale of H

            det_H = jnp.linalg.det(H)
            diff = jnp.linalg.norm(H - H_prev) / jnp.linalg.norm(H_prev)
            norm_H = jnp.linalg.norm(H)

            Lambda_TrF, sigma = self.eval(_p[:batch_size//2], _w[:batch_size//2], H)

            metrics_to_log = {
                "Rel. Change": diff,
                "Norm H": norm_H,
                "det H": det_H,
                "Λ Tr F": Lambda_TrF / (2 * np.pi),
                "σ(Λ Tr F)": sigma / (2 * np.pi),
            }

            log_parts = [f"{name}: {value.item():.4e}" for name, value in metrics_to_log.items()]
            summary_str = f"Iter {i+1}/{iterations} -> " + ", ".join(log_parts)

            tqdm.write(summary_str)

            progress.set_postfix({
                "Relative change": f"{diff.item():.4e}",
            }, refresh=False)
            key, _ = jax.random.split(key)

        return H

class HarmonicForm(HarmonicBundle):

    def __init__(self, metric_fn, H_metric_fn, monomials, coefficients, cy_dim, ambient, 
                 defining_polys=None):
        super().__init__(metric_fn, monomials, coefficients, cy_dim, ambient, defining_polys)
        self.H_metric_fn = H_metric_fn  # HYM metric on V
      # self.family_ids = [0,2,6,8,17,19,22,40,42,45,49]
        self.family_ids = [2,6,8,22,40,42,45,49]
        # self.n_harmonic = len(self.family_ids)
        self.n_harmonic = 1

        mbl, mbq = poly_utils.MonomialBasis(ambient, 1), poly_utils.MonomialBasis(ambient, 2)
        monomials_B = mbl.power_matrix
        monomials_C = self.monomial_basis.power_matrix
        variables = sp.symarray('z', ambient.item() + len(ambient))
        _monad_map_AG = [v**3 for v in variables[:4]]
        self.quotient_basis, ideal_generators, groebner_basis = poly_utils.get_quotient_basis(variables, _monad_map_AG, 
                                                                                   monomials_B, monomials_C)
        self.n_Vk = self.rank_B * poly_utils.dim_OXk(self.ambient, self.twisting_degree, self.monomial_basis.mod_degree)
        self.n_Ok = poly_utils.dim_OXk(self.ambient, self.twisting_degree, self.monomial_basis.mod_degree)
        self.N_sb = self.n_Vk - 1

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

    @partial(jax.jit, static_argnums=(0,))
    def H1XV_representatives(self, p):
        """
        Representatives of the $H^1(X;V)$ cohomology
        """
        # patch_idx = 0
        patch_idx = jnp.argmax(jnp.abs(math_utils.to_complex(p))[:self.rank_B])
        nu = self.del_bar_section_B(p)
        # project onto subbundle
        nu = jnp.delete(nu, patch_idx, axis=-2, assume_unique_indices=True)

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
    
    def section_combination(self, p, params):
        cubic_monomials = poly_utils.monomial_evaluate_log(p, self.mb3)
        S = self.twisted_section_basis(p)
        z_norm = jnp.sum(jnp.abs(p)**2, axis=-1)
        uts = jnp.einsum("...n, ...am->...amn", cubic_monomials, S) / jnp.expand_dims(z_norm**3, (0,1,2))
        psi = models.coeff_head_holoV(p, params, self.n_homo_coords, tuple(self.ambient), 
                                      self.N_sb, self.n_Ok)
        s = jnp.squeeze(jnp.einsum("...mn, ...amn->...a", psi[0], uts))
        return s

    def harmonic_rep(self, p, params):
        p_c = math_utils.to_complex(p)
        pb = self.pb_fn(p_c)
        xi = self.H1XV_representatives(p)  # [..., h^1_V, rank_V, cy_dim]
        correction_ambient = curvature.del_bar_z(p, self.section_combination, False, params)
        form_correction = jnp.einsum('...hai,...ji->...haj', correction_ambient, jnp.conj(pb))
        eta = xi + form_correction
        return eta

    def codifferential_eta(self, p, pb, params):
        g_inv = jnp.linalg.inv(self._metric_fn(p))  # \bar{\nu} \mu
        eta = self.harmonic_rep(p, params)  # [..., h^1_V, rank_V, cy_dim]
        del_z_eta = curvature.del_bar_z(p, self.harmonic_rep, False, params)
        del_z_eta = jnp.einsum("...havi, ...ui->...havu", del_z_eta, jnp.conj(pb))

        A = self.connection_form_V(p, self.H_metric_fn)
        A_eta_contract = jnp.einsum("...aci, ...hcj->...haji", A, eta)
        covariant_derivative_eta = del_z_eta + A_eta_contract
        codiff_eta = jnp.einsum("...vu, ...havu->...ha", g_inv, covariant_derivative_eta)
        return codiff_eta
    
    @partial(jax.jit, static_argnums=(0,))
    def objective_function(self, data, params, norm_control=False,
                           full_contraction=False, MAX_NORM=10.):
        (p, pb, w) = data
        vol_Omega = jnp.mean(w)
        codiff = vmap(self.codifferential_eta, in_axes=(0,0,None))(p, pb, params)
        codiff = jnp.squeeze(codiff)  # [..., i]

        if norm_control is True:
            codiff_norm = vmap(jnp.linalg.norm)(codiff) / self.n_harmonic  # don't squeeze
            codiff = jnp.where(jnp.expand_dims(codiff_norm, (1,2)) < MAX_NORM, codiff, 0.)

        if full_contraction is True:
            H = vmap(self.H_metric_fn)(p)
            integrand = jnp.einsum("...ab, ...ha, ...hb->...", H, codiff, jnp.conj(codiff))
            integrand = jnp.squeeze(integrand)
            return jnp.mean(jnp.abs(integrand) * w) / vol_Omega
        
        abs_codiff = jnp.mean(jnp.abs(codiff), axis=-1)
        # abs_codiff = jnp.where(abs_codiff < MAX_NORM, abs_codiff, 0.)
        loss = jnp.mean(abs_codiff * w) / vol_Omega
        return loss
    
    @staticmethod
    @jit
    def inner_product_Hodge(data, eta, g_pred, H_pred):
        p, weights, dVol_Omega = data
        g_inv = jnp.linalg.inv(g_pred)  # g^{\bar{\nu} \mu}

        integrand = jnp.einsum("...vu, ...mav, ...nbu, ...ab->...mn", g_inv, eta, jnp.conj(eta), H_pred)
        
        det_g = jnp.squeeze(jnp.real(jnp.linalg.det(g_pred)))
        vol_g = jnp.mean(det_g * weights / dVol_Omega)
        _weights = jnp.expand_dims(det_g * weights / dVol_Omega, axis=(1,2))
        return jnp.mean(integrand * _weights, axis=0) / vol_g
    
    def loss_breakdown(self, data, params):

        p, pb, w = data
        vol_Omega = jnp.mean(w)
        loss = self.objective_function(data, params)
        eta = vmap(self.harmonic_rep, in_axes=(0,None))(p, params)
        codiff = vmap(self.codifferential_eta, in_axes=(0,0,None))(p, pb, params)
        codiff = jnp.squeeze(codiff)
        codiff_integrand = jnp.einsum("...ab, ...ha, ...hb->...", H, codiff, jnp.conj(codiff))
        codiff_integrand = jnp.squeeze(codiff_integrand)

        g = vmap(self._metric_fn)(p)
        g_inv = jnp.linalg.inv(g)
        F = vmap(self.curvature_form_V, in_axes=(0,None))(p, self.H_metric_fn)
        H = vmap(self.H_metric_fn)(p)
        G_matter = self.inner_product_Hodge(data, eta, g, H)

        F_up = jnp.einsum("...ji, ...kl, ...abik->...abjl", g_inv, g_inv, F) #  F^{\bar{\nu} \mu}^a_b
        F_sq = jnp.einsum("...abij, ...cdij, ...db, ...ac->...", F, jnp.conjugate(F_up), jnp.linalg.inv(H), H)
        ym_energy_integrand = F_sq / 2.

        g_tr_F = jnp.einsum("...vu, ...abuv->...ab", g_inv, F)
        det_g_tr_F = jnp.linalg.det(g_tr_F)
        max_eig = vmap(jnp.linalg.norm)(g_tr_F)
        Tr_F_g = vmap(jnp.trace)(g_tr_F)

        return {'loss': loss, 'Tr_F_g': jnp.mean(w * Tr_F_g) / vol_Omega, "max_eig": jnp.mean(w * max_eig) / vol_Omega,
                'det_F_g': jnp.mean(w * det_g_tr_F) / vol_Omega, "det_H": jnp.mean(w * jnp.linalg.det(H)) / vol_Omega,
                'Tr_F_g_var': jnp.var(jnp.abs(g_tr_F)), 'codiff_norm': jnp.mean(w * jnp.abs(codiff_integrand)) / vol_Omega,
                'YM energy': jnp.mean(w * ym_energy_integrand) / vol_Omega, 'G_Kahler': jnp.diag(G_matter)}
    
    def fit(self, data_path, epochs: int = 128, batch_size: int = 512, lr: float = 1e-4,
            shuffle_rng = np.random.default_rng(), name = None,
            ckpt: dict = None):

        self.name = f"harmonic_HYM_{datetime.now().strftime('%Y-%m-%d_%H')}" if name is None else name
        self.eval_interval = 1  # epochs
        self.save_interval = 4
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

        coeff_class = models.CoeffNetwork_spectral_nn_CICY_holoV
        bundle_harmonic_model = coeff_class(self.n_homo_coords, self.ambient, self.n_units_harmonic,
                n_1=self.N_sb, n_2=self.n_Ok, n_harmonic=self.n_harmonic)

        _params, _opt_state, _ = create_train_state(_k, bundle_harmonic_model, _tx, data_dim=self.n_homo_coords * 2)
        if ckpt is not None:
            _params, _opt_state = utils.load_ckpt(_params, _opt_state, ckpt['params'], ckpt['opt'])
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(_params))
        logger.info(f'Params (Count: {param_count})=========>>>')
        logger.info(jax.tree_util.tree_map(lambda x: x.shape, _params))
        logger.info(bundle_harmonic_model.tabulate(_k, jnp.ones([1, self.n_homo_coords * 2])))

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

