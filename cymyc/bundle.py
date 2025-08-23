r"""Bundle
"""
import warnings
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, jacfwd, vmap

import os, time, tqdm
from datetime import datetime

import numpy as np

import optax
from typing import List, Callable

import cymyc.dataloading as dataloading
from cymyc.calabi_yau import CICY, Points, RicciFlatMetric

from cymyc.approx import models
from cymyc.approx.train import create_train_state
from cymyc.approx import hym
from cymyc.approx import harmonic_bundle

from cymyc.utils import math_utils
from cymyc.utils import gen_utils as utils
import cymyc.curvature as curvature

from collections import defaultdict


class HYMLineBundle:
    def __init__(self, key, cy: CICY, coefficients, metric_fn: RicciFlatMetric, tx: optax.GradientTransformation, 
                 pb_fn, n_units: List[int] = [48, 48, 48, 48], name = None):
        self._cy = cy
        self._line_bundle = coefficients
        self._slope = None  # sum(coefficients)+0.0 # TODO: Compute slope correctly
        self._metric_fn = metric_fn
        self._pb_fn = pb_fn
        self._tx = tx
        self._section_dim = None
        self._n_units = n_units

        # probably change later
        self.name = f"HYM_{datetime.now().strftime('%Y-%m-%d_%H')}" if name is None else name
        self.periodic_eval = True
        self.eval_interval = 1  # epochs
        self.save_interval = 8
        self.eval_interval_t = 512  # iterations

        self._beta_fn = models.LearnedVector_spectral_nn_CICY(
            dim     = cy.n_ambient_coords,
            ambient = cy.ambient,
            n_units = n_units,
            use_spectral_embedding = True)

        self._params, self._opt_state, _ = create_train_state(
            key, self._beta_fn, tx, data_dim = cy.n_ambient_coords*2)

        self.log_H_ref_fn = partial(hym.reference_hermitian_structure, line_bundle=self._line_bundle, 
                                    ambient=tuple(self._cy.ambient))
        self.trace_F_fn = partial(hym.trace_F, curvature_form_fn=self.curvature_form, metric_fn=self._metric_fn)
        self.degrees = self.generate_basis()
        self.n_monomials = np.prod([d.shape[0] for d in self.degrees]).item()  # number of monomials for line bundle
        self.line_bundle_monomials = self.compute_line_bundle_monomials()

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, p, params):
        # return log of Hermitian line bundle metric H
        beta = self._beta_fn.apply({'params': params}, p)
        log_H_ref = self.log_H_ref_fn(p)
        # H_fn = lambda x: H_ref_fn(x)*jnp.exp(self._beta_fn.apply({"params": self._params}, x))
        # return jax.vmap(jax.jit(H_fn))(p)
        return beta + log_H_ref

    def generate_basis(self):
        warnings.warn("Not implemented yet! Hardcoded O(1,1,0,-2)")
        degrees_1 = np.stack([np.asarray([0,1]), np.asarray([1,0])], axis=0)
        degrees_2 = np.stack([np.asarray([0,1]), np.asarray([1,0])], axis=0)
        degrees_3 = np.stack([np.asarray([0,0])], axis=0)
        degrees_4 = np.stack([np.asarray([0,-2]), np.asarray([-1,-1]), np.asarray([-2,0])], axis=0)

        return [degrees_1, degrees_2, degrees_3, degrees_4]

    def compute_line_bundle_monomials(self):
        import itertools
        degrees = self.generate_basis()
        X = list(itertools.product(*degrees))
        monomials = np.vstack([np.hstack(x) for x in X])
        return monomials
    
    def line_bundle_section_eval(self, p):
        return jnp.prod(jnp.power(p, self.line_bundle_monomials), axis=-1)

    def get_sections_fn(self):
        degrees = self.generate_basis()
        X = jnp.meshgrid(*map(lambda x: jnp.arange(x.shape[0]), degrees))
        monoms = jnp.concat([*map(lambda dx: dx[0][dx[1]], zip(degrees, X))], axis=-1).reshape(-1, 8)
        return jax.jit(lambda z: jnp.prod(jnp.power(z, monoms), axis=-1))

    @property
    def get_section_dim(self):
        if self._section_dim is not None:
            return self._section_dim
        self._section_dim = jnp.prod(jnp.asarray([*map(lambda x: x.shape[0], self.generate_basis())]))
        return self._section_dim


    def connection_form(self, p, pullbacks, params):
        A = curvature.del_z(p, self.__call__, params)
        return jnp.einsum("...a,...ia->...i", A, pullbacks)

    @partial(jax.jit, static_argnums=(0,))
    def curvature_form(self, p, pullbacks, params):
        ddbar_log_H = curvature.del_z_bar_del_z(p, self.__call__, params, wide=True)
        ddbar_log_H_pb = jnp.einsum("...ia,...jb,...ab->...ij", pullbacks, jnp.conj(pullbacks), ddbar_log_H)
        return ddbar_log_H_pb

    def callback(self, val_data, params, storage, logger, epoch, t0, slope: float = None):
        
        loss_breakdown_dict = hym.loss_breakdown(
            val_data, params, self.curvature_form, self._metric_fn, slope = slope)
        loss_breakdown_dict = jax.device_get(loss_breakdown_dict)
        summary = jax.tree_util.tree_map(lambda x: x.item(), loss_breakdown_dict)

        mode = 'VAL'
        logs = [f"{k}: {v:.4f}" for (k,v) in summary.items()]        
        logger.info(f"[{time.time()-t0:.1f}s]: [{mode}] | Epoch: {epoch}" + ''.join([f" | {log}" for log in logs]))

        [storage[k].append(v) for (k,v) in summary.items()]
        utils.save_logs(storage, self.name, epoch)
        return storage

    def load_params(self, params_path):
        _k = jax.random.key(42)
        model_class = models.LearnedVector_spectral_nn_CICY
        bundle_model = model_class(cy.n_ambient_coords.item(), self._cy.ambient, self._n_units)
        _params, init_rng = utils.random_params(_k, bundle_model, data_dim=self._cy.n_ambient_coords * 2)
        bundle_params = utils.load_params(_params, params_path)  # parameters for trained metric NN
        self._params = bundle_params
        return bundle_params


    #def fit(self, train_pts: Points, val_pts: Points = None, epochs: int = 10, batch_size: int = 128, 
    def fit(self, data_path, epochs: int = 24, batch_size: int = 128, 
            shuffle_rng = np.random.default_rng()):
        
        import jax.profiler
        # Normalize slope
        #vol = jnp.mean(train_pts.w)
        #if self._slope is not None: self._slope *= vol

        #train_loader = dataloading.DataLoader(
        #    (train_pts.pts, train_pts.pullbacks, train_pts.w), batch_size = batch_size, shuffle_rng = shuffle_rng)
        #val_loader = dataloading.DataLoader(
        #    (val_pts.pts, val_pts.pullbacks, val_pts.w), batch_size = batch_size, shuffle_rng = shuffle_rng)
        #dataset_size = train_pts.w.shape[0]
        
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

        t0 = time.time()
        with jax.default_device(device):
            for epoch in range(epochs):
                # train_loader.reset(shuffle_rng = shuffle_rng)

                if epoch % self.eval_interval == 0: 
                    # val_data = val_loader.get_val_data()
                    val_loader, val_data = dataloading.get_validation_data(val_loader, batch_size, A_val, shuffle_rng)
                    p, w, _ = val_data
                    pb = vmap(self._pb_fn)(math_utils.to_complex(p))
                    val_data = (p, pb, w)
                    storage = self.callback(
                        val_data, self._params, storage, logger, epoch, t0, self._slope)

                if epoch > 0: 
                    train_loader = dataloading.data_loader(A_train, batch_size, shuffle_rng)

                wrapped_train_loader = tqdm.tqdm(train_loader, desc=f'Epoch {epoch}', total=dataset_size//batch_size, 
                                            colour='green', mininterval=0.1)

                # with tqdm.tqdm(train_loader, desc=f"Epoch: {epoch}", mininterval=0.1, colour="green") as data_iter:
                    # for t, data in enumerate(data_iter):
                global_step = 0
                for t, data in enumerate(wrapped_train_loader):
                    # with jax.profiler.StepTraceAnnotation("train_step", step_num=global_step):
                    p, w, _ = data
                    pb = vmap(self._pb_fn)(math_utils.to_complex(p))
                    data = (p, pb, w)

                    self._params, self._opt_state, loss = hym.train_step(
                        data, self._params, self._opt_state, self._tx, self.curvature_form, self._metric_fn, 
                        self._slope)
                    # data_iter.set_description(f"epoch: {epoch+1}/{epochs}, loss: {loss:.5f}", refresh=False)
                    wrapped_train_loader.set_postfix_str(f"loss: {loss:.5f}", refresh=False)

                    if t % self.eval_interval_t == 0:
                        storage["train_loss"].append(loss)
                    # global_step += 1
                    # if global_step > 20: break
        return storage


class TQBundle:
    @staticmethod
    @jax.jit
    def reference_form_nu(p, r_vec, cdtype = jnp.complex64):
        z = math_utils.to_complex(p).astype(cdtype)

        monoms = jnp.asarray([
            z[0]*z[2], # x0 y0
            z[1]*z[2], # x1 y0
            z[0]*z[3], # x0 y1
            z[1]*z[3]]) # x1 y1
        R = r_vec @ monoms

        # nu_coeff = ((jnp.abs(z[-2:])**2).sum(axis=-1)**(-2))
        nu_coeff = 1/jnp.sum(z[-2:]*jnp.conj(z[-2:]))**2
        mu = z[6]*(jnp.zeros((8,)).at[7].set(1)) - z[7]*(jnp.zeros((8,)).at[6].set(1))

        nu = R*nu_coeff * jnp.conj(mu)
        return nu



class Harmonic:
    def __init__(self, key, cy, representative_fn, metric: RicciFlatMetric, bundle: HYMLineBundle, 
                 tx: optax.GradientTransformation, pb_fn: Callable, n_units: List[int] = [48, 48, 48, 48], name = None):
        self._representative_fn = representative_fn
        self._metric = metric
        self._bundle = bundle
        self._cy = cy
        self._pb_fn = pb_fn
        self._tx = tx
        self._n_units = n_units

        self.name = f"harmonic_bundle_{datetime.now().strftime('%Y-%m-%d_%H')}" if name is None else name
        self.periodic_eval = True
        self.eval_interval = 1  # epochs
        self.save_interval = 8
        self.eval_interval_t = 512  # iterations

        self._model = models.LearnedVector_spectral_nn(
            dim = cy.n_ambient_coords,
            ambient = cy.ambient,
            n_units = n_units,
            n_out = 2*self._bundle.n_monomials,
            use_spectral_embedding=True)

        self._params, self._opt_state, _ = create_train_state(
            key, self._model, self._tx, data_dim = cy.n_ambient_coords*2)
        
        self._monomials = self._bundle.compute_line_bundle_monomials()

        ## strip later
        self.ambient = self._cy._ambient
        self.degrees = self.ambient + 1
        self.n_hyper = len(self.degrees)
        self.proj_idx = jnp.asarray(utils._generate_proj_indices(self.degrees))
        self.bounds = jnp.cumsum(jnp.concatenate((jnp.zeros(1), self.degrees)))
        self.n_transitions = utils._patch_transitions(self.n_hyper, len(self.ambient), self.degrees)
        self.n_ambient = len(self.ambient)

    def __call__(self, p, params, pullbacks):
        # return $\nu = \nu_0 + \overline{\partial}_V s$.
        nu_0 = self._representative_fn(p)
        del_bar_V_s = curvature.del_bar_z(p, self.line_bundle_section, params)
        nu = nu_0 + del_bar_V_s

        return jnp.einsum('...a, ...ia->...i', nu, jnp.conj(pullbacks))

    @partial(jax.jit, static_argnums=(0,))
    def line_bundle_section(self, p, params):
        # evaluate section of line bundle
        p_c = math_utils.to_complex(p)
        coeffs = self._model.apply({'params': params}, p)
        coeffs = math_utils.to_complex(coeffs)
        mono_eval = jnp.prod(jnp.power(p_c, self._monomials), axis=-1)
        return jnp.sum(coeffs * mono_eval, axis=-1)

    def contract_nu_g(self, p, params, pullbacks):
        g = self._metric(p)
        g_inv = jnp.linalg.inv(g)  # \bar{\mu} \nu
        bundle_form = self.__call__(p, params, pullbacks)
        return jnp.einsum("...u,...uv->...v", bundle_form, g_inv)  # [..., (s), i]

    def bundle_codiff(self, p, params, pullbacks):
        # essentially trace of covariant derivative
        pb = pullbacks
        A = self._bundle.connection_form(p, pb, self._bundle._params)  # [..., (s), i, (t)] s,t line bundle idx
        nu_g_contraction = self.contract_nu_g(p, params, pb)  # [..., (s), i]

        d_contraction = curvature.del_z(p, self.contract_nu_g, params, pb)  # [... (s), i, a]
        d_contraction = jnp.einsum("...ia,...ja->...ij", d_contraction, pb)  # [..., (s), i, j]

        Tr_d_contraction = jnp.einsum("...ii->...", d_contraction)
        Tr_connection = jnp.einsum("...i, ...i->...", A, nu_g_contraction)
        codiff = Tr_d_contraction + Tr_connection
        return Tr_d_contraction + Tr_connection
    
    def objective_function(self, data, params, norm_order=1., MAX_CODIFF_NORM=100.):
        p, pb, w = data
        integrand = vmap(self.bundle_codiff, in_axes=(0,None,0))(p, params, pb)
        integrand = jnp.where(jnp.abs(integrand) < MAX_CODIFF_NORM, integrand, 0.)
        return jnp.mean(w * jnp.abs(integrand) ** norm_order) / jnp.mean(w)

    def loss_breakdown(self, data, params, norm_order=1.):
        p, pb, w = data
        loss = self.objective_function(data, params, norm_order)

        g = vmap(self._metric)(p)
        g_inv = jnp.linalg.inv(g)
        nu_0 = vmap(self._representative_fn)(p)
        del_bar_V_s = vmap(curvature.del_bar_z, in_axes=(0,None,None))(p, self.line_bundle_section, params)

        nu_0 = jnp.einsum('...a, ...ia->...i', nu_0, jnp.conj(pb))
        del_bar_V_s = jnp.einsum('...a, ...ia->...i', del_bar_V_s, jnp.conj(pb))
        ref_norm = jnp.einsum('...ij, ...i, ...j-> ...', g_inv, nu_0, jnp.conj(nu_0))
        correction_norm = jnp.einsum('...ij, ...i, ...j-> ...', g_inv, del_bar_V_s, jnp.conj(del_bar_V_s))

        return {'loss': loss, 'ref_norm': jnp.mean(w * jnp.abs(ref_norm) ** norm_order) / jnp.mean(w),
                'corr_norm': jnp.mean(w * jnp.abs(correction_norm) ** norm_order) / jnp.mean(w)}

    def create_dataloader(self, pts, batch_size, shuffle = np.random.default_rng()) -> dataloading.DataLoader:
        g = self._metric(pts.pts)
        ref_pb = jnp.einsum("...i,...ai->...a",
            jax.vmap(self._representative_fn)(pts.pts), jnp.conj(pts.pullbacks))[..., jnp.newaxis, :]
        dref_pb = jnp.einsum(
            "...ij,...ai,...bj->...ab",
            jax.vmap(partial(del_z, fun = self._representative_fn))(pts.pts),
            jnp.conj(pts.pullbacks), pts.pullbacks)
        A = self._bundle.connection_form(pts.pts, pts.pullbacks)
        d_dual_ref = -jnp.einsum("...ba,...ab->...", jnp.linalg.inv(g), dref_pb + (A[..., jnp.newaxis]*ref_pb))

        return DataLoader(
            (pts.pts,         # Points
             pts.pullbacks,   # Pullbacks
             pts.w,           # Weights
             g,               # Metric
             A,               # Connection form
             d_dual_ref),     # d+reference
            batch_size = batch_size, shuffle_rng = shuffle)

    @partial(jax.jit, static_argnums=(0,4))
    def train_step(self, data, params, opt_state, optimizer):
        loss, grads = jax.value_and_grad(self.objective_function, argnums=1)(data, params)
        param_updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, param_updates)
        return params, opt_state, loss

    def get_section_eval_fn(self):
        def section(pts, params):
            s_basis_fn = self._bundle.get_sections_fn()

            s_fn = lambda p: self._model.apply({"params": params}, p)
            n = self._bundle.get_section_dim
            s_vals = s_fn(pts)

            s = (s_vals[..., :n] + 1j*s_vals[..., n:])*s_basis_fn(math_utils.to_complex(pts))
            return s.sum(axis=-1)
        return jax.jit(section)

    def callback(self, val_data, params, storage, logger, epoch, t0):

        loss_breakdown_dict = self.loss_breakdown(val_data, params)
        loss_breakdown_dict = jax.device_get(loss_breakdown_dict)
        summary = jax.tree_util.tree_map(lambda x: x.item(), loss_breakdown_dict)

        mode = 'VAL'
        logs = [f"{k}: {v:.4f}" for (k,v) in summary.items()]        
        logger.info(f"[{time.time()-t0:.1f}s]: [{mode}] | Epoch: {epoch}" + ''.join([f" | {log}" for log in logs]))

        [storage[k].append(v) for (k,v) in summary.items()]
        utils.save_logs(storage, self.name, epoch)
        return storage
    
    # def fit(self, train_pts: Points, epochs: int = 10, batch_size: int = 128, shuffle_rng = np.random.default_rng()):
    def fit(self, data_path, epochs: int = 10, batch_size: int = 128, shuffle_rng = np.random.default_rng()):
        

        storage = defaultdict(list)
        logger = utils.logger_setup(self.name, filepath=os.path.abspath(__file__))
        data_path = os.path.join(data_path, 'dataset.npz')
        # train_loader = self.create_dataloader(train_pts, batch_size, shuffle_rng)

        A_train, A_val, train_loader, val_loader, psi = dataloading.initialize_loaders_train(shuffle_rng, data_path, 
            batch_size, logger=logger)
        dataset_size = A_train[0].shape[0]

        try:
            device = jax.devices('gpu')[0]
        except:
            print("Unable to use gpu, falling back to cpu")
            device = jax.devices('cpu')[0]

        # s_fn = self.get_section_eval_fn()
        t0 = time.time()

        with jax.default_device(device):
            for epoch in range(epochs):
                # train_loader.reset(shuffle_rng = shuffle_rng)

                if epoch % self.eval_interval == 0:
                    val_loader, val_data = dataloading.get_validation_data(val_loader, batch_size, A_val, shuffle_rng)
                    p, w, _ = val_data
                    pb = vmap(self._pb_fn)(math_utils.to_complex(p))
                    val_data = (p, pb, w)
                    storage = self.callback(val_data, self._params, storage, logger, epoch, t0)

                if epoch > 0: 
                    train_loader = dataloading.data_loader(A_train, batch_size, shuffle_rng)

                wrapped_train_loader = tqdm.tqdm(train_loader, desc=f'Epoch {epoch}', total=dataset_size//batch_size, 
                                            colour='green', mininterval=0.1)
                
                # with tqdm.tqdm(train_loader, desc=f"Epoch: {epoch}", mininterval=0.1, colour="green") as data_iter:
                #     for t, data in enumerate(data_iter):
                for t, data in enumerate(wrapped_train_loader):
                    p, w, _ = data
                    pb = vmap(self._pb_fn)(math_utils.to_complex(p))
                    data = (p, pb, w)
                    self._params, self._opt_state, loss = self.train_step(
                        data, self._params, self._opt_state, self._tx)
                    
                    wrapped_train_loader.set_postfix_str(f"loss: {loss:.5f}", refresh=False)
                    # data_iter.set_description(f"epoch: {epoch+1}/{epochs}, loss: {loss:.5f}", refresh=False)
                    if t % self.eval_interval_t == 0:
                        storage["train_loss"].append(loss)
        return storage

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
    
    def transition_loss(self, p, params=None, norm=1., max_jac_norm=10.):
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

        return p_repeated, p_transformed

if __name__ == "__main__":
    import argparse
    from cymyc.calabi_yau import DworkQuintic, TQ
    from jax.profiler import trace

    description = "CY metric learning."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ds", "--dataset", type=str, default=None, help="Path to dataset.")
    parser.add_argument("-m_ckpt", "--metric_ckpt", type=str, default=None, help="Flat metric checkpoint.")

    cmd_args = parser.parse_args()
    seed = int(time.time()) # 42
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)

    # cy = DworkQuintic(0.1)
    cy = TQ(psi_vals = [2., 1.])

    n_epochs, batch_size = 1, 1024
    n_units = [48, 48, 48, 48]

    key = jax.random.key(0)
    key, _k = jax.random.split(key)
    
    if cmd_args.dataset is not None:
        pts = cy.load_points(data_path=cmd_args.dataset)
    else:
        n_pts = 500000
        pts = cy.sample_points(key = _k, max_pts = n_pts)

    Metric = RicciFlatMetric(_k, cy, tx = optax.adamw(1e-4),
                             n_units = n_units)
    if cmd_args.metric_ckpt is not None:
        print('Loading metric!')
        from flax import linen as nn
        # initialize model
        if (cy.n_hyper > 1) or (len(cy.ambient) > 1):
            model_class = models.LearnedVector_spectral_nn_CICY
        else:
            model_class = models.LearnedVector_spectral_nn

        # load metric model
        g_model = model_class(cy.n_ambient_coords, cy.ambient, n_units)
        _params, init_rng = utils.random_params(init_rng, g_model, data_dim=cy.n_ambient_coords * 2)
        g_params = utils.load_params(_params, cmd_args.metric_ckpt)  # parameters for trained metric NN
        g_FS_fn, g_correction_fn, pb_fn = models.helper_fns(cy)
        metric = jax.tree_util.Partial(models.ddbar_phi_model, params=g_params, 
                                       g_ref_fn=g_FS_fn, g_correction_fn=g_correction_fn)
    else:
        print('Training metric!')
        key, _k = jax.random.split(key)
        Metric.fit(pts, epochs=n_epochs, batch_size=batch_size)
        utils.basic_ckpt(metric._params, metric._opt_state, 'test_flat_metric', 'FIN')
        metric = Metric.__call__

    # line_bundle = tuple((1,))
    line_bundle = tuple([1, 1, 0, -2])

    key, _k = jax.random.split(key)
    bundle = HYMLineBundle(
        _k, cy, line_bundle, metric, optax.adamw(1e-4),
        pb_fn = Metric._pb_fn,
        n_units = n_units, name = "HYM_test")
    
    profile_dir = "/tmp/jax-trace-bundle-tensorboard"
    os.makedirs(profile_dir, exist_ok=True) # Ensure the directory exists
    # --- start JAX Profiler Trace ---
    # with trace(profile_dir):
    print('Training bundle!')
    storage = bundle.fit(cmd_args.dataset, epochs=n_epochs, batch_size=batch_size)
    # --- End JAX Profiler Trace ---
    print(f"Bundle fit profile saved to: {profile_dir}")
    # storage = bundle.fit(pts, pts, epochs=n_epochs, batch_size=batch_size)
    utils.basic_ckpt(bundle._params, bundle._opt_state, 'test_hym_bundle', 'FIN')

    import matplotlib.pyplot as plt
    import mplhep as hep; hep.style.use('CMS')

    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111)
    ax.plot(storage['train_loss'])
    plt.savefig("hym_loss.pdf", bbox_inches='tight')

    loader = dataloading.DataLoader(
        (pts.pts, pts.pullbacks, pts.w), batch_size = 512)

    F_vals = []
    for _data in tqdm.tqdm(loader):
        g_tr_F = bundle.trace_F_fn(_data, bundle._params)
        F_vals.append(g_tr_F)

    F_vals = jnp.asarray(F_vals)
    F_vals = jnp.real(F_vals.reshape(-1)) / pts.w.mean()
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111)
    _ = ax.hist(F_vals, bins=100)
    plt.savefig("hym_hist.pdf", bbox_inches='tight')

    print("Mean:", F_vals.mean())
