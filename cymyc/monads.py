import jax

import numpy as np  # original CPU-backed NumPy
import jax.numpy as jnp

from jax import jit, jacfwd, vmap, random
import optax

from functools import partial

import math, time, argparse, os
import sympy as sp

import time
from tqdm import tqdm

# custom
from cymyc.utils import math_utils, poly_utils
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

    def __init__(self, monomials, coefficients, cy_dim, ambient, metric_fn, defining_polys=None):
        
        self.ambient = ambient
        self.ambient_dim = sum(self.ambient)
        self.cy_dim = cy_dim
        self.metric_fn = metric_fn

        # specify monad data
        # make arguments later
        # self.family_ids = [0,2,6,8,17,19,22,40,42,45,49]
        self.family_ids = [2,6,8,22,40,42,45,49]

        self.n_harmonic = len(self.family_ids)
        self.rank_V = 3
        self.twisting_degree = 4
        self.line_bundle_B = (1,1,1,1)
        self.rank_B = len(self.line_bundle_B)
        self.line_bundle_C = (4,)
        self.mb3 = jnp.asarray(poly_utils.monomial_basis(ambient, 3)) # for basis of sections of $V \otimes O_X(k)$
        self.mb4 = jnp.asarray(poly_utils.monomial_basis(ambient, 4)) # for untwisting sections
        self.cdtype = np.complex64

        self.n_hyper = self.ambient_dim - self.cy_dim
        self.n_homo_coords = monomials.shape[-1]
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
            self.monomial_basis = poly_utils.MonomialBasisReduced(ambient, self.twisting_degree, defining_polys, psi)

        self.all_mono_eval_fn = jax.tree_util.Partial(poly_utils.monomial_evaluate_log, 
                                                      s_k=self.monomial_basis.power_matrix, 
                                                      conj=False)

        self.n_Vk = self.rank_B * poly_utils.dim_OXk(self.ambient, self.twisting_degree-1, self.monomial_basis.mod_degree)
        self.n_Ok = poly_utils.dim_OXk(self.ambient, self.twisting_degree, self.monomial_basis.mod_degree)
        
        mbl, mbq = poly_utils.MonomialBasis(ambient, 1), poly_utils.MonomialBasis(ambient, 2)
        variables = sp.symarray('z', ambient.item() + len(ambient))
        monad_map = [v**3 for v in variables[:4]]
        self.monad_map_power_matrix = poly_utils.monomials_to_power_matrix(monad_map, variables)
        monomials_B = mbl.power_matrix
        monomials_C = self.monomial_basis.power_matrix
        self.quotient_basis, ideal_generators, groebner_basis = poly_utils.get_quotient_basis(variables, monad_map, 
                                                                                   monomials_B, monomials_C)
        
        self.eps_3d = jnp.array(math_utils.n_dim_eps_symbol(3))

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
        
    def embedding_matrix(self, p, cdtype=np.complex64):
        r"""
        Describes the embedding $\iota: V \righthookarrow B$.
        """
        patch_idx = jnp.argmax(jnp.abs(math_utils.to_complex(p))[:self.rank_B])
        proj = jnp.eye(self.rank_V, dtype=cdtype)
        f_p = poly_utils.monomial_evaluate_log(p, self.monad_map_power_matrix)
        col = -f_p / f_p[patch_idx]
        col = jnp.delete(col, patch_idx, assume_unique_indices=True)
        return jnp.insert(proj, patch_idx, col, axis=-1)

    def section_basis(self, p):
        """
        Section basis described in ambient coordinates, expressed in a local
        frame $Z_i$ on $U_i$. 
        """
        _sections = self.embedding_matrix(p)
        return _sections
          
    
    def twisted_section_basis(self, p, cdtype=np.complex64):
        r"""
        Holomorphic sections of twisted dual bundle $V^{\vee} \otimes O_X(k)$,
        expressed in a local frame - typically Z_i^k. 
        """
        p_c = math_utils.to_complex(p)
        patch_idx = jnp.argmax(jnp.abs(p_c)[:self.rank_B])
        
        # section_matrix = jnp.zeros((self.rank_B, len(self.mb3) * self.rank_B), dtype=cdtype)
        Ok_powers = self.mb3.at[:,patch_idx].subtract(3)
        Ok_monomials = poly_utils.monomial_evaluate_log(p, Ok_powers)
    
        blocks = [Ok_monomials] * self.rank_B
        section_matrix = jax.scipy.linalg.block_diag(*blocks)
        embedding_matrix = self.embedding_matrix(p, cdtype)
    
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
        pb = self.pb_fn(math_utils.to_complex(p))
        A = hym.connection_form_V(p, pb, self.section_metric_network, params)
        return A

    @partial(jax.jit, static_argnums=(0,))
    def curvature_form(self, p, params):
        pb = self.pb_fn(math_utils.to_complex(p))
        F = hym.curvature_form_V(p, pb, self.section_metric_network, params)
        return F

    def section_network_VOk(self, p, params, activation=nn.gelu):
        r"""
        Returns a smooth section of $V$ from linear combination of twisted holomorphic basis 
        elements for $V^{\vee} \otimes O_X(k)$.
        """
        p_c = math_utils.to_complex(p)
        H_inv = jnp.linalg.inv(self.fubini_study_metric_V(p))  # H^{\bar{\nu} \mu}
    
        n_homo_coords_i = np.array(self.ambient) + 1  # coords for each ambient space factor 
    
        # (n_h, n_Vk, n_Ok) * n_A if all ambient space factors identical
        # TODO
        coeffs = models.coeff_head_holoV(p, params, self.n_homo_coords, tuple(self.ambient), self.n_Vk, self.n_Ok,
                                         self.n_harmonic, activation)
        coeffs = [jnp.ones_like(c) for c in coeffs]
        V_section = jnp.empty((self.n_harmonic, self.cy_dim,), dtype=self.cdtype)
        
        for i in range(len(self.ambient)):
            s, e = int(np.sum(self.ambient[:i]) + i), int(np.sum(self.ambient[:i+1]) + i + 1)
            n_c, as_idx = e-s, jnp.triu_indices(e-s,1)
            p_ambient_i = jax.lax.dynamic_slice(p_c, (s,), (n_c,))
    
            # Basis of twisted one-forms in ambient P^1 x ... x P^n
            Z_i = p_ambient_i  # coords on the i-th ambient space
            Z_norm_sq_i = jnp.sum(jnp.abs(Z_i)**2)
            Ok_monomials = poly_utils.monomial_evaluate_log(p, self.mb4)
            _V_section_twisted_i = self.twisted_section_basis(p) #TODO

            _V_section_i = jnp.einsum("...hxi, ...ax, ...ab, ...i->...hb", coeffs[i], 
                                      jnp.conjugate(_V_section_twisted_i), H_inv, Ok_monomials) / (Z_norm_sq_i ** 4)                                
            V_section += _V_section_i
    
        return V_section  # don't squeeze

    def section_metric_network(self, p, params, activation=nn.gelu):
        r"""
        Returns a smooth section of $Sym(V^* \otimes V^*$) from basis of sections for $V$.
        """
        p_c = math_utils.to_complex(p)
        H_fs = self.fubini_study_metric_B(p)
        
        # (n_h, n_Vk, n_Ok) * n_A if all ambient space factors identical
        # TODO
        coeffs = models.coeff_head_holoV(p, params, self.n_homo_coords, tuple(self.ambient), self.rank_V, 
                                         self.rank_V, 1, complex_kernel=False, activation=activation)
        sym_2B_section = jnp.empty((self.rank_B, self.rank_B), dtype=self.cdtype)
        
        for i in range(len(self.ambient)):
            s, e = int(np.sum(self.ambient[:i]) + i), int(np.sum(self.ambient[:i+1]) + i + 1)
            n_c = e-s
            p_c_ambient_i = jax.lax.dynamic_slice(p_c, (s,), (n_c,))
    
            # Basis of twisted one-forms in ambient P^1 x ... x P^n
            Z_i = p_c_ambient_i  # coords on the i-th ambient space
            Z_norm_sq_i = jnp.sum(jnp.abs(Z_i)**2)
            _V_section = self.section_basis(math_utils.to_real(Z_i))  # [rank_V, dim]
            _V_dual_section = jnp.einsum("...ab, ...ib->...ia", H_fs, jnp.conjugate(_V_section))
            sym1 = jnp.einsum("...ia, ...jb->...ijab", _V_dual_section, jnp.conjugate(_V_dual_section))
            sym2 = jnp.einsum("...ia, ...jb->...jiab", _V_dual_section, jnp.conjugate(_V_dual_section))
            matrix_update = jnp.einsum("...ij, ...ijab->...ab", jnp.logaddexp(jnp.squeeze(coeffs[i]), 0), 
                                       0.5 * (sym1 + sym2))
            sym_2B_section += matrix_update
    
        metric_ansatz = H_fs + sym_2B_section
        return jnp.einsum("...ab, ...ia, ...jb->...ij", metric_ansatz, _V_section, 
                          jnp.conjugate(_V_section))
    
    def fit(self, data_path, epochs: int = 24, batch_size: int = 128, lr: float = 1e-4,
            shuffle_rng = np.random.default_rng()):

        storage = defaultdict(list)
        logger = utils.logger_setup(self.name, filepath=os.path.abspath(__file__))
        data_path = os.path.join(data_path, 'dataset.npz')
        # cmd_args.metadata_path = os.path.join(cmd_args.dataset, 'metadata.pkl')

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

        _tx = optax.adamw(lr)
        self.n_units_harmonic = [48, 48, 48, 48]
        model_class = models.CoeffNetwork_spectral_nn_CICY_holoV
        bundle_metric_model = model_class(self.n_homo_coords, self.ambient, self.n_units_harmonic, n_1=self.rank_V,
                                           n_2=self.rank_V, n_harmonic=1, activation=nn.gelu)
        _params, _opt_state, _ = create_train_state(
            _k, bundle_metric_model, _tx, data_dim=self.n_homo_coords * 2)


        t0 = time.time()
        with jax.default_device(device):
            for epoch in range(epochs):

                if epoch % self.eval_interval == 0: 
                    val_loader, val_data = dataloading.get_validation_data(val_loader, batch_size, A_val, shuffle_rng)
                    p, w, _ = val_data
                    pb = vmap(self._pb_fn)(math_utils.to_complex(p))
                    val_data = (p, pb, w)
                    storage = self.callback(
                        val_data, _params, storage, logger, epoch, t0, self._slope)

                if epoch > 0: 
                    train_loader = dataloading.data_loader(A_train, batch_size, shuffle_rng)

                wrapped_train_loader = tqdm.tqdm(train_loader, desc=f'Epoch {epoch}', total=dataset_size//batch_size, 
                                            colour='green', mininterval=0.1)

                global_step = 0
                for t, data in enumerate(wrapped_train_loader):
                    p, w, _ = data
                    pb = vmap(self._pb_fn)(math_utils.to_complex(p))
                    data = (p, pb, w)

                    _params, _opt_state, loss = hym.train_step(
                        data, _params, _opt_state, _tx, self.curvature_form, self._metric_fn, 
                        self._slope)
                    wrapped_train_loader.set_postfix_str(f"loss: {loss:.5f}", refresh=False)

                    if t % self.eval_interval_t == 0:
                        storage["train_loss"].append(loss)
                    # global_step += 1
                    # if global_step > 20: break
        return storage
