"""
Geometric computations on complex structure moduli space.
"""

import os, multiprocessing
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count())
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, jacfwd, vmap, random

import numpy as np
import typing
from functools import partial

from tqdm import tqdm

from typing import List, Callable, Mapping, Tuple
from jaxtyping import Array, Float, Complex, ArrayLike

# custom
from .. import alg_geo, fubini_study
from ..utils import math_utils


# TODO Register as pytree.
# Be careful not to mutate the class after initialization. This will either trigger
# recompilation (and be very slow), or result in unexpected behaviour! This will be fixed soon...
class WP:
    def __init__(self, cy_dim: int, monomials: typing.List[np.array], ambient: np.array):
        jax.config.update("jax_enable_x64", True)

        self.cy_dim = cy_dim
        self.monomials = monomials
        self.ambient = ambient
        self.n_homo_coords = monomials[0].shape[1]
        self.n_hyper = len(monomials)  # number of defining polys
        self.n_projective = len(ambient)  # number of projective space factors
        self.dim = np.sum(ambient) - self.n_hyper  # dimension of fibre (CY)
        self.degrees = self.ambient + 1
        # sum of inhomo coords for each projective space
        self.n_inhomo_coords = sum(self.degrees) - len(self.degrees)
        self.conf_mat, p_conf_mat = math_utils._configuration_matrix(monomials, ambient) 
        self.t_degrees = math_utils._find_degrees(self.ambient, self.n_hyper, self.conf_mat)
        self.kmoduli_ambient = math_utils._kahler_moduli_ambient_factors(self.cy_dim, self.ambient, self.t_degrees)

        self.interior_product_sgn = np.asarray([(-1)**(k+1) for k in range(self.dim)])
        self.b_product_sgn = np.asarray([[(-1)**((j+1)-(i+1)-1) for i in range(self.dim)] for j in range(self.dim)])
        self.cdtype = np.complex64

        self.eps_3d = jnp.array(math_utils.n_dim_eps_symbol(3))

        if (self.n_hyper > 1) or (len(self.ambient) > 1):
            self.integration_weights_fn = partial(alg_geo._integration_weights_cicy,
                n_hyper=self.n_hyper, cy_dim=self.dim, n_coords=self.n_homo_coords,
                ambient=self.ambient, kmoduli_ambient=self.kmoduli_ambient, cdtype=self.cdtype)
        else:
            self.integration_weights_fn = partial(alg_geo.compute_integration_weights,
                cy_dim=self.dim)

    def _fs_metric_total(self, p, normalization=jax.lax.complex(1.,0.), cdtype=np.complex64):
        r"""
        Returns ambient FS metric evaluated in product of projective spaces,
        P^{k_1}_1 \times P^{k_2}_2 \times \cdots \times P^{k_n}_n,
        returned in inhomogeneous coordinates.
        Parameters
        ----------
            `p`     : Complex `n+1`-dim homogeneous coords at 
                      which metric matrix is evaluated. Shape [i].
        Return
        """
        g_FS = jnp.zeros((self.n_inhomo_coords, self.n_inhomo_coords), dtype=cdtype)
        for i in range(len(self.ambient)):
            pt_s, pt_e = np.sum(self.ambient[:i]) + i, np.sum(self.ambient[:i+1]) + i + 1
            g_s, g_e = np.sum(self.ambient[:i]), np.sum(self.ambient[:i+1])

            p_ambient_i = jax.lax.dynamic_slice(p, (pt_s,), (pt_e-pt_s,))
            p_ambient_i_inhomo = math_utils._inhomogenize(p_ambient_i)
            g_FS_ambient_i = fubini_study._fs_metric(p_ambient_i_inhomo, normalization, cdtype)
            g_FS = jax.lax.dynamic_update_slice(g_FS, g_FS_ambient_i, (g_s, g_s))
    
        return g_FS

    def _fs_metric_inverse_total(self, p, normalization=jax.lax.complex(1.,0.), cdtype=np.complex64):
        r"""
        Returns ambient FS metric inverse evaluated in product of projective spaces,
        P^{k_1}_1 \times P^{k_2}_2 \times \cdots \times P^{k_n}_n,
        returned in inhomogeneous coordinates.
        Parameters
        ----------
            `p`     : Complex `n+1`-dim homogeneous coords at 
                      which metric matrix is evaluated. Shape [i].
        Return
        """
        g_FS_inv = jnp.zeros((self.n_inhomo_coords, self.n_inhomo_coords), dtype=cdtype)
        for i in range(len(self.ambient)):
            pt_s, pt_e = np.sum(self.ambient[:i]) + i, np.sum(self.ambient[:i+1]) + i + 1
            g_s, g_e = np.sum(self.ambient[:i]), np.sum(self.ambient[:i+1])

            p_ambient_i = jax.lax.dynamic_slice(p, (pt_s,), (pt_e-pt_s,))
            p_ambient_i_inhomo = math_utils._inhomogenize(p_ambient_i)
            g_FS_inv_ambient_i = fubini_study._fs_metric_inverse(p_ambient_i_inhomo, normalization, cdtype)
            g_FS_inv = jax.lax.dynamic_update_slice(g_FS_inv, g_FS_inv_ambient_i, (g_s, g_s))
            
        return g_FS_inv
    
    def _compute_dQdz_inhomo(self, p, dQdz_monomials, dQdz_coeffs):
        """
        Compute Jacobians then projects to inhomogeneous coords,
        returns dQdz, shape [n_inhomo_coords, n_hyper]
        """
        dQdz_all = jnp.stack([alg_geo.evaluate_poly(p, dm, dc) for (dm, dc) in 
                    zip(dQdz_monomials, dQdz_coeffs)], axis=-1)
        ones_mask = jnp.logical_not(jnp.isclose(p, jax.lax.complex(1.,0.)))
        good_dQdz = dQdz_all[jnp.nonzero(ones_mask, size=self.n_inhomo_coords),:]

        if self.n_hyper == 1: return jnp.expand_dims(jnp.squeeze(good_dQdz), axis=-1)
        return jnp.squeeze(good_dQdz)

    @partial(jit, static_argnums=(0,2))
    def zeta_jacobian(self, p, deformation, dQdz_monomials, dQdz_coeffs):
        """
        Outputs vector field in ambient space corresponding to diffeomorphism between
        fibres of Kuranishi family. 
        """
        x = math_utils.to_real(p)

        dim = p.shape[-1]  # complex dimension
        real_jac_x = jacfwd(self.zeta)(x, deformation, dQdz_monomials, dQdz_coeffs)
        dzeta_dx = real_jac_x[..., :dim]
        dzeta_dy = real_jac_x[..., dim:]
        dzeta_dz = 0.5 * (dzeta_dx - 1.j * dzeta_dy)
        dzeta_dzbar = 0.5 * (dzeta_dx + 1.j * dzeta_dy)

        return dzeta_dz, dzeta_dzbar

    def zeta(self, x, deformation, dQdz_monomials, dQdz_coeffs):

        p = math_utils.to_complex(x)

        g_FS_inv = jnp.conjugate(self._fs_metric_inverse_total(p)) # g^{\mu \bar{\nu}}
        dQdz = self._compute_dQdz_inhomo(p, dQdz_monomials, dQdz_coeffs)  # in the full ambient space
        H = jnp.einsum('...ij, ...ia, ...jb->...ab', g_FS_inv, dQdz, jnp.conjugate(dQdz))
        
        cm_def = deformation(p)
        dphi_dt = -jnp.einsum('...ab,...uv,...va,...b->...u', jnp.linalg.inv(H), g_FS_inv, 
                              jnp.conjugate(dQdz), cm_def)

        return dphi_dt
    
    def _zeta(self, x, deformations, dQdz_monomials, dQdz_coeffs):
        """
        Multiple deformations (all $h^{2,1}$ of them).
        """
        p = math_utils.to_complex(x)

        g_FS_inv = jnp.conjugate(self._fs_metric_inverse_total(p)) # g^{\mu \bar{\nu}}
        dQdz = self._compute_dQdz_inhomo(p, dQdz_monomials, dQdz_coeffs)  # in the full ambient space
        H = jnp.einsum('...ij, ...ia, ...jb->...ab', g_FS_inv, dQdz, jnp.conjugate(dQdz))
        
        cm_deformations = deformations(p)  # [h^{(2,1)}, n_hyper] 
        dphi_dt = -jnp.einsum('...ab,...uv,...va,...nb->...nu', jnp.linalg.inv(H), g_FS_inv, 
                              jnp.conjugate(dQdz), cm_deformations)

        return dphi_dt

    
    @partial(jit, static_argnums=(0,))
    def project_to_good_CY_coords(self, A, mask):
        return A[jnp.nonzero(mask,size=self.dim),:]

    @partial(jit, static_argnums=(0,))
    def project_to_good_ambient_coords(self, A, mask):
        return A[jnp.nonzero(mask,size=self.n_inhomo_coords),:]

    @partial(jit, static_argnums=(0,))
    def compute_bij(self, p, dzeta_dzbar, pb, ones_mask, dQdz_monomials, dQdz_coeffs):
        """
        (2,1) form, parameterized by (i,k) - for the missing dz^i factor and d\bar{z}^{\bar{k}}.
        """
        dQdz_homo = alg_geo.evaluate_dQdz(p, dQdz_monomials, dQdz_coeffs)
        _, good_coord_mask = alg_geo.argmax_dQdz_cicy(p, dQdz_homo, self.n_hyper, self.n_homo_coords, True)
        good_coord_mask = good_coord_mask[jnp.nonzero(ones_mask, size=self.n_inhomo_coords)]#.reshape(self.n_inhomo_coords)
        
        # jacobian of diffeomorphism in fibre coordinates w.r.t. homogeneous coords in ambient space
        dzeta_dzbar_fibre = self.project_to_good_CY_coords(dzeta_dzbar, good_coord_mask)
        # now pullback d\bar{z}^{\mu} to CY and take the interior product with \Omega. The coefficient
        # factor is accounted for by multiplication by the weights later.
        dzeta_dzbar_fibre_pb = jnp.einsum('i,...ia,...ja->...ij', self.interior_product_sgn, 
                                         dzeta_dzbar_fibre, jnp.conjugate(pb))
        return dzeta_dzbar_fibre_pb


    @partial(jit, static_argnums=(0,4))
    def compute_wp_metric_diagonal(self, p, dQdz_monomials, dQdz_coeffs, deformation=None):
        """
        Computes diagonal element of WP metric obtained by MC integration over fibre points `p`.
        Tangent vector defined by `deformation` argument. See Mirror Symmetry, Mori, eq. (6.1).
        NB: Don't `vmap` this.
        """

        if p.dtype in (np.float32, np.float64):
            p = math_utils.to_complex(p)

        weights, pb, *_ = vmap(self.integration_weights_fn, in_axes=(0,None,None))(
            p, dQdz_monomials, dQdz_coeffs)

        if (self.n_hyper == 1) and (len(self.ambient) == 1):
            dQdz_monomials, dQdz_coeffs = [dQdz_monomials], [dQdz_coeffs]
            
        vol_Omega = jnp.mean(weights)
        ones_mask = jnp.logical_not(jnp.isclose(p, jax.lax.complex(1.,0.)))

        dzeta_dz, dzeta_dzbar = vmap(self.zeta_jacobian, in_axes=(0,None,None,None))(p, deformation, dQdz_monomials, dQdz_coeffs)

        bij = vmap(self.compute_bij, in_axes=(0,0,0,0,None,None))(p, dzeta_dzbar, pb, ones_mask, dQdz_monomials, dQdz_coeffs)
        bb = jnp.einsum('ij,...ij,...ji->...', self.b_product_sgn, bij, jnp.conjugate(bij))

        _a = vmap(self.project_to_good_ambient_coords)(jnp.einsum('...ij->...ji',dzeta_dz), ones_mask)
        a = -jnp.squeeze(jnp.einsum('...ii->...', _a))
        aa = a * jnp.conjugate(a)

        weights, a, aa, bb = jnp.squeeze(weights), jnp.squeeze(a), jnp.squeeze(aa), jnp.squeeze(bb)
        int_a = jnp.mean(weights * a)

        g_wp = -jnp.mean(weights * (aa + bb)) / vol_Omega + (jnp.conjugate(int_a) * int_a) / vol_Omega**2
        return g_wp, vol_Omega
    
    # @partial(jit, static_argnums=(0,6))
    def _compute_wp_metric_diagonal_batch_i(self, p, weights, pb, dQdz_monomials, dQdz_coeffs, deformation=None):
        """
        Computes diagonal elements of WP metric obtained by MC integration over fibre points `p`. 
        Tangent vector defined by `deformation` argument. See Mirror Symmetry, Mori, eq. (6.1).
        Yields mean over batch, to be processed using incremental averaging.
        NB: Don't `vmap` this.
        """
    
        ones_mask = jnp.logical_not(jnp.isclose(p, jax.lax.complex(1.,0.)))
        dzeta_dz, dzeta_dzbar = vmap(self.zeta_jacobian, in_axes=(0,None,None,None))(p, deformation, dQdz_monomials, dQdz_coeffs)

        bij = vmap(self.compute_bij, in_axes=(0,0,0,0,None,None))(p, dzeta_dzbar, pb, ones_mask, dQdz_monomials, dQdz_coeffs)
        bb = jnp.einsum('ij,...ij,...ji->...', self.b_product_sgn, bij, jnp.conjugate(bij))

        _a = vmap(self.project_to_good_ambient_coords)(jnp.einsum('...ij->...ji',dzeta_dz), ones_mask)
        a = -jnp.squeeze(jnp.einsum('...ii->...', _a))
        aa = a * jnp.conjugate(a)

        weights, a, aa, bb = jnp.squeeze(weights), jnp.squeeze(a), jnp.squeeze(aa), jnp.squeeze(bb)
        vol_Omega_i, int_a_i, int_aa_p_bb_i = jnp.mean(weights), jnp.mean(weights * a), jnp.mean(weights * (aa + bb))

        return vol_Omega_i, int_a_i, int_aa_p_bb_i

    def compute_wp_batched_diagonal(self, data, monomials, coefficients, deformation):
        # online batch calculation
        vol_Omega, int_a, int_aa_p_bb = 0., 0., 0.
        n = 0

        dQdz_info = [alg_geo.dQdz_poly(self.n_homo_coords, m, c) for (m,c) in zip(monomials, coefficients)]
        dQdz_monomials, dQdz_coeffs = list(zip(*dQdz_info))
        _data_gen = lambda: zip(*data)

        for t, data in enumerate(tqdm(_data_gen(), leave=False)):
            p, w, pb = data
            B = p.shape[0]
            n += B

            _vol_Omega, _int_a, _int_aa_p_bb = self._compute_wp_metric_diagonal_batch_i(p, w, pb, dQdz_monomials, dQdz_coeffs, deformation)
            vol_Omega = math_utils.online_update(vol_Omega, _vol_Omega, n, B)
            int_a = math_utils.online_update(int_a, _int_a, n, B)
            int_aa_p_bb = math_utils.online_update(int_aa_p_bb, _int_aa_p_bb, n, B)
        
        g_wp = -int_aa_p_bb / vol_Omega + (jnp.conjugate(int_a) * int_a) / vol_Omega**2
        return g_wp, vol_Omega
    

    def pullback_diffeo_jacobian(self, p, dzeta_dzbar, pb, ones_mask, dQdz_homo):
        """
        Returns (0,1)-T_X-valued form.
        """
        _, good_coord_mask = alg_geo.argmax_dQdz_cicy(p, dQdz_homo, self.n_hyper, self.n_homo_coords, True)
        good_coord_mask = good_coord_mask[jnp.nonzero(ones_mask, size=self.n_inhomo_coords)]#.reshape(self.n_inhomo_coords)
        
        # jacobian of diffeomorphism in fibre (CY) coordinates w.r.t. homogeneous coords in ambient space
        dzeta_dzbar_fibre = self.project_to_good_CY_coords(dzeta_dzbar, good_coord_mask)
        # now pullback d\bar{z}^{\mu} to CY - no interior product with Omega!
        dzeta_dzbar_fibre_pb = jnp.einsum('...ia,...ja->...ij', dzeta_dzbar_fibre, jnp.conjugate(pb))
        return dzeta_dzbar_fibre_pb

    @partial(jit, static_argnums=(0,4,5,6))
    def yukawas(self, p, dQdz_monomials, dQdz_coeffs, deformation_a, 
                deformation_b, deformation_c, weights=None, pb=None):
        r"""
        Yukawa couplings for a CY threefold. $\frac{d^3 \Omega}{ds^a ds^b ds^c} \vert_{s=0}$.
        """

        if weights is None:
            weights, pb, *_ = vmap(self.integration_weights_fn, in_axes=(0,None,None))(
                p, dQdz_monomials, dQdz_coeffs)
        
        ones_mask = jnp.logical_not(jnp.isclose(p, jax.lax.complex(1.,0.)))
        dQdz_homo = vmap(alg_geo.evaluate_dQdz, in_axes=(0,None,None))(p, dQdz_monomials, dQdz_coeffs)
        Omega = vmap(alg_geo._holomorphic_volume_form, in_axes=(0,0,None,None,None))(
            p, dQdz_homo, self.n_hyper, self.n_homo_coords, self.ambient)

        # get KS representative along diffeomorphisms r,s
        _, dzeta_a_dzbar = vmap(self.zeta_jacobian, in_axes=(0,None,None,None))(
            p, deformation_a, dQdz_monomials, dQdz_coeffs)
        # pullback to CY
        dzeta_a_dzbar_pb = vmap(self.pullback_diffeo_jacobian)(p, dzeta_a_dzbar, pb, 
                ones_mask, dQdz_homo)

        if (deformation_a == deformation_b) and (deformation_b == deformation_c):
            # simplify for one-parameter families
            dzeta_b_dzbar_pb = dzeta_a_dzbar_pb
            dzeta_c_dzbar_pb = dzeta_a_dzbar_pb
        else:
            _, dzeta_b_dzbar = vmap(self.zeta_jacobian, in_axes=(0,None,None,None))(
                p, deformation_b, dQdz_monomials, dQdz_coeffs) 
            _, dzeta_c_dzbar = vmap(self.zeta_jacobian, in_axes=(0,None,None,None))(
                p, deformation_c, dQdz_monomials, dQdz_coeffs)   
        
            dzeta_b_dzbar_pb = vmap(self.pullback_diffeo_jacobian)(p, dzeta_b_dzbar, pb, 
                    ones_mask, dQdz_homo)
            dzeta_c_dzbar_pb = vmap(self.pullback_diffeo_jacobian)(p, dzeta_c_dzbar, pb, 
                    ones_mask, dQdz_homo)
        
        contraction = jnp.einsum('...ijk, ...xyz, ...ix, ...jy, ...kz -> ...', 
                   self.eps_3d, self.eps_3d, dzeta_a_dzbar_pb, dzeta_b_dzbar_pb, 
                   dzeta_c_dzbar_pb)
        contraction = jnp.squeeze(contraction)

        kappa_abc = Omega**2 * contraction
        dVol_Omega = Omega * jnp.conjugate(Omega)

        return jnp.mean(weights * kappa_abc / dVol_Omega)

    def compute_yukawas_batched(self, data, monomials, coefficients, deformation_a, deformation_b, deformation_c):
        kappa, n = 0., 0
        dQdz_info = [alg_geo.dQdz_poly(self.n_homo_coords, m, c) for (m,c) in zip(monomials, coefficients)]
        dQdz_monomials, dQdz_coeffs = list(zip(*dQdz_info))
        _data_gen = lambda: zip(*data)

        for t, data in enumerate(tqdm(_data_gen(), leave=False)):
            p, w, pb = data
            B = p.shape[0]
            n += B
            _kappa = self.yukawas(p, dQdz_monomials, dQdz_coeffs, deformation_a, 
                deformation_b, deformation_c, w, pb)
            
            kappa = math_utils.online_update(kappa, _kappa, n, B)

        return kappa


# TODO Register as pytree
class WP_full(WP):
    # Handles multiple deformations, i.e. tangent vectors in M_{cs}
    def __init__(self, cy_dim: int, monomials: List[np.array], ambient: ArrayLike, 
                 deformations: List[Callable]):
        r"""Base class for geometric computations over complex structure moduli space.

        Parameters
        ----------
        cy_dim : int
            Dimension of Calabi-Yau manifold.
        monomials : List[np.array]
            List of defining monomials.
        ambient : ArrayLike
            Dimensions of the ambient space factors.
        deformations : List[Callable]
            List of functions representing complex structure deformations.

        Notes
        -----
        Here the `deformations` parameter is a list of polynomial deformations corresponding 
        to independent tangent vectors of the complex structure moduli space - there should be 
        $h^{(2,1)}$ deformations to construct the complete moduli space metric. For example, for 
        the deformation family of the intersection of two cubics in $\mathbb{P}^5$,

        $$
        B_{\psi} = \left\{\begin{array}{c}Z_0^3 + Z_1^3 + Z_2^3 - 3 \psi Z_3 Z_4 Z_5 = 0\\
        Z_3^3 + Z_4^3 + Z_5^3 - 3 \psi Z_0 Z_1 Z_2 = 0\end{array} \, : \, \psi \in \mathbb{C}\right\} \subset \mathbb{P}^5~.
        $$

        The single complex structure moduli direction corresponds to the trilinear polynomial deformations above, 
        and we can write down this deformation explicitly:

        ```python
        def X33_deformation(p, precision=np.complex128):
            d1 = jnp.einsum("...a,aj->...j", jnp.expand_dims(p[3]*p[4]*p[5], axis=-1),
                            jnp.asarray([[-3.,0.]], precision))
            d2 = jnp.einsum("...a,aj->...j", jnp.expand_dims(p[0]*p[1]*p[2], axis=-1),
                            jnp.asarray([[0.,-3.]], precision))
            return d1 + d2
        ```
        """
        super().__init__(cy_dim, monomials, ambient)
        self.cdtype = jnp.complex128
        self.h_21 = len(deformations)
        self.deformations = deformations
        self.deformation_indices = jnp.arange(self.h_21)

    @partial(jit, static_argnums=(0,2))
    def zeta_jacobian(self, p, deformation_idx, dQdz_monomials, dQdz_coeffs):

        print(f'Compiling {self.zeta_jacobian.__qualname__}, deformation={deformation_idx}')
        x = math_utils.to_real(p)

        dim = p.shape[-1]  # complex dimension
        real_jac_x = jacfwd(self.zeta)(x, deformation_idx, dQdz_monomials, dQdz_coeffs)
        dzeta_dx = real_jac_x[..., :dim]
        dzeta_dy = real_jac_x[..., dim:]
        dzeta_dz = 0.5 * (dzeta_dx - 1.j * dzeta_dy)
        dzeta_dzbar = 0.5 * (dzeta_dx + 1.j * dzeta_dy)

        return dzeta_dz, dzeta_dzbar


    def zeta(self, x, deformation_idx, dQdz_monomials, dQdz_coeffs):

        p = math_utils.to_complex(x)

        g_FS_inv = jnp.conjugate(self._fs_metric_inverse_total(p, cdtype=self.cdtype)) # g^{\mu \bar{\nu}}
        dQdz = self._compute_dQdz_inhomo(p, dQdz_monomials, dQdz_coeffs)
        H = jnp.einsum('...ij, ...ia, ...jb->...ab', g_FS_inv, dQdz, jnp.conjugate(dQdz))

        deformation = self.deformations[deformation_idx]
        cm_def = deformation(p)
        dphi_dt = -jnp.einsum('...ab,...uv,...va,...b->...u', jnp.linalg.inv(H), g_FS_inv,
                              jnp.conjugate(dQdz), cm_def)

        return dphi_dt

    def deformation_vector(self, idx, p):
        return jax.lax.switch(idx, self.deformations, p)

    @partial(jit, static_argnums=(0,))
    def zeta_jacobian_complete(self, p, dQdz_monomials, dQdz_coeffs):

        print(f'Compiling {self.zeta_jacobian_complete.__qualname__}')
        x = math_utils.to_real(p)

        dim = p.shape[-1]  # complex dimension
        real_jac_x = jacfwd(self.zeta_complete)(x, dQdz_monomials, dQdz_coeffs)
        dzeta_dx = real_jac_x[..., :dim]
        dzeta_dy = real_jac_x[..., dim:]
        dzeta_dz = 0.5 * (dzeta_dx - 1.j * dzeta_dy)
        dzeta_dzbar = 0.5 * (dzeta_dx + 1.j * dzeta_dy)

        return dzeta_dz, dzeta_dzbar


    def zeta_complete(self, x, dQdz_monomials, dQdz_coeffs):

        p = math_utils.to_complex(x)

        g_FS_inv = jnp.conjugate(self._fs_metric_inverse_total(p, cdtype=self.cdtype)) # g^{\mu \bar{\nu}}
        dQdz = self._compute_dQdz_inhomo(p, dQdz_monomials, dQdz_coeffs)
        H = jnp.einsum('...ij, ...ia, ...jb->...ab', g_FS_inv, dQdz, jnp.conjugate(dQdz))

        # deformation = self.deformations[deformation_idx]
        # cm_def = deformation(p)
        cm_defs = vmap(self.deformation_vector, in_axes=(0,None))(self.deformation_indices, p)  # all h^{(2,1)} deformations
        dphi_dt = -jnp.einsum('...ab,...uv,...va,...cb->...cu', jnp.linalg.inv(H), g_FS_inv,
                              jnp.conjugate(dQdz), cm_defs)

        return dphi_dt  # [..., h_{21}, n_inhomo_coords]

    def compute_wp_metric_rs_component(self, p, dQdz_monomials, dQdz_coeffs, deformation_r_idx, deformation_s_idx):
        """
        Computes (r,s) component of WP metric obtained by MC integration over fibre points `p`.
        Tangent vectors defined by `deformation_*` arguments. See Mirror Symmetry, Mori, eq. (6.1).
        NB: Don't `vmap` this.
        """

        weights, pb, *_ = vmap(self.integration_weights_fn, in_axes=(0,None,None))(
            p, dQdz_monomials, dQdz_coeffs)
        vol_Omega = jnp.mean(weights)
        ones_mask = jnp.logical_not(jnp.isclose(p, jax.lax.complex(1.,0.)))

        # get Lie derivative of Ω along diffeomorphisms r,s
        dzeta_r_dz, dzeta_r_dzbar = vmap(self.zeta_jacobian, in_axes=(0,None,None,None))(p, deformation_r_idx, dQdz_monomials, dQdz_coeffs)
        dzeta_s_dz, dzeta_s_dzbar = vmap(self.zeta_jacobian, in_axes=(0,None,None,None))(p, deformation_s_idx, dQdz_monomials, dQdz_coeffs)

        Br = vmap(self.compute_bij, in_axes=(0,0,0,0,None,None))(p, dzeta_r_dzbar, pb, ones_mask, dQdz_monomials, dQdz_coeffs)
        Bs = vmap(self.compute_bij, in_axes=(0,0,0,0,None,None))(p, dzeta_s_dzbar, pb, ones_mask, dQdz_monomials, dQdz_coeffs)
        Br_w_Bs = jnp.einsum('ij,...ij,...ji->...', self.b_product_sgn, Br, jnp.conjugate(Bs))

        _Ar = vmap(self.project_to_good_ambient_coords)(jnp.einsum('...ij->...ji',dzeta_r_dz), ones_mask)
        _As = vmap(self.project_to_good_ambient_coords)(jnp.einsum('...ij->...ji',dzeta_s_dz), ones_mask)
        Ar = -jnp.einsum('...ii->...', _Ar)
        As = -jnp.einsum('...ii->...', _As)
        Ar_w_As = Ar * jnp.conjugate(As)

        int_Ar = jnp.mean(weights * Ar)
        int_As = jnp.mean(weights * As)

        G_wp_rs = -jnp.mean(weights * (Ar_w_As + Br_w_Bs)) / vol_Omega + (jnp.conjugate(int_As) * int_Ar) / vol_Omega**2
        return G_wp_rs

    
    # NOTE: Recompiles for every new deformation passed iniitally if `jit`'ed - should only `jit` if
    # scanning across a large grid
    @partial(jit, static_argnums=(0,6,7))
    def _compute_wp_rs_batch(self, p, weights, pb, dQdz_monomials, dQdz_coeffs, r, s):
        """
        Computes (r,s) element of WP metric obtained by MC integration over fibre points `p`. 
        Tangent vectors defined by r,s indices arguments. See Mirror Symmetry, Mori, eq. (6.1).
        Yields mean over batch, to be processed using incremental averaging.
        NB: Don't `vmap` this.
        """
        ones_mask = jnp.logical_not(jnp.isclose(p, jax.lax.complex(1.,0.)))
        
        # get Lie derivative of Ω along diffeomorphisms r,s
        dzeta_r_dz, dzeta_r_dzbar = vmap(self.zeta_jacobian, in_axes=(0,None,None,None))(p, r, dQdz_monomials, dQdz_coeffs)
        dzeta_s_dz, dzeta_s_dzbar = vmap(self.zeta_jacobian, in_axes=(0,None,None,None))(p, s, dQdz_monomials, dQdz_coeffs)

        Br = vmap(self.compute_bij, in_axes=(0,0,0,0,None,None))(p, dzeta_r_dzbar, pb, ones_mask, dQdz_monomials, dQdz_coeffs)
        Bs = vmap(self.compute_bij, in_axes=(0,0,0,0,None,None))(p, dzeta_s_dzbar, pb, ones_mask, dQdz_monomials, dQdz_coeffs)
        Br_w_Bs = jnp.einsum('ij,...ij,...ji->...', self.b_product_sgn, Br, jnp.conjugate(Bs))

        _Ar = vmap(self.project_to_good_ambient_coords)(jnp.einsum('...ij->...ji',dzeta_r_dz), ones_mask)
        _As = vmap(self.project_to_good_ambient_coords)(jnp.einsum('...ij->...ji',dzeta_s_dz), ones_mask)
        Ar = -jnp.einsum('...ii->...', _Ar)
        As = -jnp.einsum('...ii->...', _As)
        Ar_w_As = Ar * jnp.conjugate(As)

        weights, Ar, As, Ar_w_As, Br_w_Bs = jnp.squeeze(weights), jnp.squeeze(Ar), jnp.squeeze(As), jnp.squeeze(Ar_w_As), jnp.squeeze(Br_w_Bs)
        vol_Omega_i, int_Ar_i, int_As_i, int_AB_i = jnp.mean(weights), jnp.mean(weights * Ar), jnp.mean(weights * As), jnp.mean(weights * (Ar_w_As + Br_w_Bs))

        return vol_Omega_i, int_Ar_i, int_As_i, int_AB_i

    def compute_wp_rs_component_batched(self, data, dQdz_monomials, dQdz_coeffs, r, s):
        # online batch calculation
        n = 0
        vol_Omega, int_Ar, int_As, int_AwA_p_BwB = 0., 0., 0., 0.
        _data_gen = lambda: zip(*data)

        for t, data in enumerate(_data_gen()):
            p, w, pb = data
            B = p.shape[0]
            n += B
            _vol_Omega, _int_Ar, _int_As, _int_AwA_p_BwB = self._compute_wp_rs_batch(p, w, pb, dQdz_monomials, dQdz_coeffs, r, s)

            vol_Omega = math_utils.online_update(vol_Omega, _vol_Omega, n, B)
            int_Ar = math_utils.online_update(int_Ar, _int_Ar, n, B)
            int_As = math_utils.online_update(int_As, _int_As, n, B)
            int_AwA_p_BwB = math_utils.online_update(int_AwA_p_BwB, _int_AwA_p_BwB, n, B)
        
        G_wp_rs = -int_AwA_p_BwB / vol_Omega + (jnp.conjugate(int_As) * int_Ar) / vol_Omega**2
        return G_wp_rs, vol_Omega

    def wp_metric_full_batched(self, data, monomials, coefficients, dim_moduli_cs):

        dQdz_info = [alg_geo.dQdz_poly(self.n_homo_coords, m, c) for (m,c) in zip(monomials, coefficients)]
        dQdz_monomials, dQdz_coeffs = list(zip(*dQdz_info))

        G_wp = np.zeros(2 * (dim_moduli_cs,), self.cdtype)
        for r in tqdm(range(dim_moduli_cs)):
            for s in tqdm(range(r+1)):
                G_wp[r,s], vol_Omega = self.compute_wp_rs_component_batched(data, dQdz_monomials, dQdz_coeffs, r, s)

        return G_wp, vol_Omega

    @partial(jit, static_argnums=(0,))
    def compute_wp_metric_complete(self, p: Float[Array, "... i"], dQdz_monomials: List[np.array], 
                                   dQdz_coeffs: List[np.array]) -> Complex[Array, "h_21 h_21"]:
        r"""Computes the full $h^{2,1} \times h^{2,1}$ metric $\mathcal{G}_{a\overline{b}}$ over complex structure moduli space 
        (the Weil-Petersson metric). This is obtained by Monte Carlo integration over the fibres of the deformation 
        family. Letting $(-,-)$ denote the standard intersection pairing on $H^{p,q}_{\overline{\partial}}(X)$ with $p+q=n$:

        $$
        \begin{align}
        \mathcal{G}_{a\bar{b}} =
        \left(\frac{d\Omega_t}{d  t^a}, \frac{d\Omega_t}{d t^b}\right)\bigg\vert_{t_0} - \frac{1}{(\Omega,\Omega)}\left(\Omega, \frac{d\Omega_t}{d t^a}\right)\bigg\vert_{t_0} 
        \cdot  \overline{\left(\Omega, \frac{d\Omega_t}{d t^b}\right)}\bigg\vert_{t_0}\,.
        \end{align}
        $$

        Note that the number of integration points required is exponential in the dimension of
        moduli space. See the article [arxiv:2401.15078](https://arxiv.org/abs/2401.15078) and Mirror symmetry, Mori, 
        eq. (6.1). for more details.
        NB: Don't `vmap` this.

        Parameters
        ----------
        p : Float[Array, "... i"]  
            2 * `complex_dim` real coords at which `fun` is evaluated. Note batch indices.
        dQdz_monomials : List[np.array]
            List of monomials corresponding to polynomial Jacobian $dQ/dz$.
        dQdz_coeffs : List[np.array]
            List of coefficients corresponding to polynomial Jacobian $dQ/dz$.

        Returns
        -------
        G_wp: Complex[Array, "h_21 h_21"]
            Weil-Petersson metric at the point `p` in complex structure moduli space.

        Notes
        -----
        Owing to vectorisation, this is significantly more efficient than computing individual components separately.
        """

        weights, pb, *_ = vmap(self.integration_weights_fn, in_axes=(0,None,None))(
            p, dQdz_monomials, dQdz_coeffs)
        vol_Omega = jnp.mean(weights)
        ones_mask = jnp.logical_not(jnp.isclose(p, jax.lax.complex(1.,0.)))

        # get Lie derivative of Ω along all diffeomorphism directions [..., h_{21}, n_inhomo_coords, n_homo_coords]
        dzeta_dz, dzeta_dzbar = vmap(self.zeta_jacobian_complete, in_axes=(0,None,None))(p, dQdz_monomials, dQdz_coeffs)

        B = vmap(vmap(self.compute_bij, in_axes=(None,0,None,None,None,None)),
                 in_axes=(0,0,0,0,None,None))(p, dzeta_dzbar, pb, ones_mask, dQdz_monomials, dQdz_coeffs)
        B = jnp.squeeze(B)
        B_w_B = jnp.einsum('ij,...aij,...bji->...ab', self.b_product_sgn, B, jnp.conjugate(B))

        _A = vmap(vmap(self.project_to_good_ambient_coords, in_axes=(0,None)), in_axes=(0,0))(
            jnp.einsum('...ij->...ji',dzeta_dz), ones_mask)
        _A = jnp.squeeze(_A)
        A = -jnp.einsum('...ii->...', _A)
        A_w_A = jnp.einsum('...a,...b->...ab', A, jnp.conjugate(A))

        int_A = jnp.mean(jnp.expand_dims(weights, axis=1) * A, axis=0)
        G_wp = -jnp.mean(jnp.expand_dims(weights, axis=(1,2)) * (A_w_A + B_w_B), axis=0) / vol_Omega + \
            jnp.einsum('...a, ...b->...ab', jnp.conjugate(int_A), int_A) / vol_Omega**2
        return G_wp

    @partial(jit, static_argnums=(0,6))
    def _compute_wp_complete_batch(self, p, weights, pb, dQdz_monomials, dQdz_coeffs, output_variance=False):
        """
        Computes (r,s) element of WP metric obtained by MC integration over fibre points `p`. 
        Tangent vectors defined by r,s indices arguments. See Mirror Symmetry, Mori, eq. (6.1).
        Yields mean over batch, to be processed using incremental averaging.
        NB: Don't `vmap` this.
        """
        ones_mask = jnp.logical_not(jnp.isclose(p, jax.lax.complex(1.,0.)))

        # get Lie derivative of Ω along all diffeomorphism directions [..., h_{21}, n_inhomo_coords, n_homo_coords]
        dzeta_dz, dzeta_dzbar = vmap(self.zeta_jacobian_complete, in_axes=(0,None,None))(p, dQdz_monomials, dQdz_coeffs)

        B = vmap(vmap(self.compute_bij, in_axes=(None,0,None,None,None,None)),
                 in_axes=(0,0,0,0,None,None))(p, dzeta_dzbar, pb, ones_mask, dQdz_monomials, dQdz_coeffs)
        B = jnp.squeeze(B)
        B_w_B = jnp.einsum('ij,...aij,...bji->...ab', self.b_product_sgn, B, jnp.conjugate(B))

        _A = vmap(vmap(self.project_to_good_ambient_coords, in_axes=(0,None)), in_axes=(0,0))(
            jnp.einsum('...ij->...ji',dzeta_dz), ones_mask)
        _A = jnp.squeeze(_A)
        A = -jnp.einsum('...ii->...', _A)
        A_w_A = jnp.einsum('...a,...b->...ab', A, jnp.conjugate(A))

        weights, A, A_w_A, B_w_B = jnp.squeeze(weights), jnp.squeeze(A), jnp.squeeze(A_w_A), jnp.squeeze(B_w_B)

        vol_Omega_i = jnp.real(jnp.mean(weights))

        wA = jnp.expand_dims(weights, axis=1) * A
        int_A_i = jnp.mean(wA, axis=0)

        wApB = jnp.expand_dims(weights, axis=(1,2)) * (A_w_A + B_w_B)
        int_AB_i = jnp.mean(wApB, axis=0)

        if output_variance is True:
            S_vol_Omega_i =  jnp.mean((weights - vol_Omega_i)**2)

            S_int_re_A_i = jnp.mean((jnp.real(wA) - jnp.real(int_A_i))**2, axis=0)
            S_int_im_A_i = jnp.mean((jnp.imag(wA) - jnp.imag(int_A_i))**2, axis=0)

            S_int_re_AB_i = jnp.mean((jnp.real(wApB) -  jnp.real(int_AB_i))**2, axis=0)
            S_int_im_AB_i = jnp.mean((jnp.imag(wApB) - jnp.imag(int_AB_i))**2, axis=0)   

            return (vol_Omega_i, int_A_i, int_AB_i), (S_vol_Omega_i, S_int_re_A_i, S_int_im_A_i, S_int_re_AB_i, S_int_im_AB_i)

        return vol_Omega_i, int_A_i, int_AB_i

    def compute_wp_complete_batched(self, data, dQdz_monomials, dQdz_coeffs, output_variance=False):
        # online batch calculation
        n, vol_Omega = 0., 0.
        int_A = jnp.zeros(self.h_21, self.cdtype)
        int_AB = jnp.zeros((self.h_21, self.h_21), self.cdtype)

        S_vol_Omega = 0.
        S_re_int_A = jnp.zeros(self.h_21, self.cdtype)
        S_im_int_A = jnp.zeros(self.h_21, self.cdtype)
        S_re_int_AB = jnp.zeros((self.h_21, self.h_21), self.cdtype)
        S_im_int_AB = jnp.zeros((self.h_21, self.h_21), self.cdtype)
        _data_gen = lambda: zip(*data)

        for t, _data in enumerate(tqdm(_data_gen(), total=len(data[0]))):
            p, w, pb = _data
            B = p.shape[0]

            if output_variance is True:
                (_vol_Omega, _int_A, _int_AB), (_S_vol_Omega_i, _S_int_re_A_i, _S_int_im_A_i, _S_int_re_AB_i, _S_int_im_AB_i) = \
                    self._compute_wp_complete_batch(p, w, pb, dQdz_monomials, dQdz_coeffs, output_variance=True)
                vol_Omega, S_vol_Omega = math_utils.online_update(vol_Omega, _vol_Omega, n, B, S_vol_Omega, _S_vol_Omega_i)

                re_int_A, S_re_int_A = math_utils.online_update_array(jnp.real(int_A), jnp.real(_int_A), n, B, S_re_int_A, _S_int_re_A_i)
                im_int_A, S_im_int_A = math_utils.online_update_array(jnp.imag(int_A), jnp.imag(_int_A), n, B, S_im_int_A, _S_int_im_A_i)
                int_A = jax.lax.complex(re_int_A, im_int_A)

                re_int_AB, S_re_int_AB = math_utils.online_update_array(jnp.real(int_AB), jnp.real(_int_AB), n, B, S_re_int_AB, _S_int_re_AB_i)
                im_int_AB, S_im_int_AB = math_utils.online_update_array(jnp.imag(int_AB), jnp.imag(_int_AB), n, B, S_im_int_AB, _S_int_im_AB_i)
                int_AB = jax.lax.complex(re_int_AB, im_int_AB)

            else:
                _vol_Omega, _int_A, _int_AB = self._compute_wp_complete_batch(p, w, pb, dQdz_monomials, dQdz_coeffs)
                vol_Omega = math_utils.online_update(vol_Omega, _vol_Omega, n, B)
                int_A = math_utils.online_update_array(int_A, _int_A, n, B)
                int_AB = math_utils.online_update_array(int_AB, _int_AB, n, B)
            n += B

        G_wp = -int_AB / vol_Omega + \
            jnp.einsum('...a, ...b->...ab', jnp.conjugate(int_A), int_A) / vol_Omega**2

        
        if output_variance is True: 
            # separate variance for real, im components - only consider diagonal
            V_re_int_A = S_re_int_A / (n-1)
            V_re_int_AB_diag = jnp.diag(S_re_int_AB) / (n-1)
            V_im_int_A = S_im_int_A / (n-1)

            V_G_wp_diag = V_re_int_AB_diag / vol_Omega**2 + \
                (4 * re_int_A**2 * V_re_int_A + \
                 4 * im_int_A**2 * V_im_int_A ) / vol_Omega**4
            return (G_wp, V_G_wp_diag), (vol_Omega, S_vol_Omega/(n-1))

        return G_wp, vol_Omega


    @partial(jit, static_argnums=(0,4,5,6))
    def kappa_abc(self, p, dQdz_monomials, dQdz_coeffs, deformation_a_idx, 
                deformation_b_idx, deformation_c_idx, weights=None, pb=None):
        r"""
        Yukawa couplings for a CY threefold. $\int_{X_s} \Omega \wedge \frac{d^3 \Omega}{ds^a ds^b ds^c} \vert_{s=0}.$
        """

        if weights is None:
            weights, pb, *_ = vmap(self.integration_weights_fn, in_axes=(0,None,None))(
                p, dQdz_monomials, dQdz_coeffs)
        
        ones_mask = jnp.logical_not(jnp.isclose(p, jax.lax.complex(1.,0.)))
        dQdz_homo = vmap(alg_geo.evaluate_dQdz, in_axes=(0,None,None))(p, dQdz_monomials, dQdz_coeffs)
        Omega = vmap(alg_geo._holomorphic_volume_form, in_axes=(0,0,None,None,None))(
            p, dQdz_homo, self.n_hyper, self.n_homo_coords, self.ambient)

        # get Lie derivative of Ω along diffeomorphisms r,s
        _, dzeta_a_dzbar = vmap(self.zeta_jacobian, in_axes=(0,None,None,None))(
            p, deformation_a_idx, dQdz_monomials, dQdz_coeffs)
        # pullback to CY
        dzeta_a_dzbar_pb = vmap(self.pullback_diffeo_jacobian)(p, dzeta_a_dzbar, pb, 
                ones_mask, dQdz_homo)

        if (deformation_a_idx == deformation_b_idx) and (deformation_b_idx == deformation_c_idx):
            # simplify for one-parameter families
            dzeta_b_dzbar_pb = dzeta_a_dzbar_pb
            dzeta_c_dzbar_pb = dzeta_a_dzbar_pb
        else:
            _, dzeta_b_dzbar = vmap(self.zeta_jacobian, in_axes=(0,None,None,None))(
                p, deformation_b_idx, dQdz_monomials, dQdz_coeffs) 
            _, dzeta_c_dzbar = vmap(self.zeta_jacobian, in_axes=(0,None,None,None))(
                p, deformation_c_idx, dQdz_monomials, dQdz_coeffs)   
        
            dzeta_b_dzbar_pb = vmap(self.pullback_diffeo_jacobian)(p, dzeta_b_dzbar, pb, 
                    ones_mask, dQdz_homo)
            dzeta_c_dzbar_pb = vmap(self.pullback_diffeo_jacobian)(p, dzeta_c_dzbar, pb, 
                    ones_mask, dQdz_homo)
        
        contraction = jnp.einsum('...ijk, ...xyz, ...ix, ...jy, ...kz -> ...', 
                   self.eps_3d, self.eps_3d, dzeta_a_dzbar_pb, dzeta_b_dzbar_pb, 
                   dzeta_c_dzbar_pb)
        contraction = jnp.squeeze(contraction)

        kappa_abc = Omega**2 * contraction
        dVol_Omega = Omega * jnp.conjugate(Omega)

        return jnp.mean(weights * kappa_abc / dVol_Omega)

    def compute_yukawa_rst_component_batched(self, data, dQdz_monomials, dQdz_coeffs, a_idx, b_idx, c_idx):
        
        kappa, n = 0., 0
        _data_gen = lambda: zip(*data)

        for t, data in enumerate(_data_gen()):
            p, w, pb = data
            B = p.shape[0]
            n += B
            _kappa = self.kappa_abc(p, dQdz_monomials, dQdz_coeffs, a_idx, b_idx, c_idx, w, pb)
            kappa = math_utils.online_update(kappa, _kappa, n, B)

        return kappa


    def yukawas_cubic_batched(self, data, monomials, coefficients, dim_moduli_cs):
        r"""
        Computes 'diagonal' Yukawas $\kappa_{aaa}$
        """
        dQdz_info = [alg_geo.dQdz_poly(self.n_homo_coords, m, c) for (m,c) in zip(monomials, coefficients)]
        dQdz_monomials, dQdz_coeffs = list(zip(*dQdz_info))

        kappa = np.zeros((dim_moduli_cs,), self.cdtype)
        for r in tqdm(range(dim_moduli_cs)):
            kappa[r] = self.compute_yukawa_rst_component_batched(data, dQdz_monomials, dQdz_coeffs, r, r, r)

        return kappa


    def yukawas_arbitrary_batched(self, data, monomials, coefficients, deformation_idx):
        r"""
        Computes Yukawas for arbitrary indices $\kappa_{abc}$ given by 'deformation_idx'.
        """
        dQdz_info = [alg_geo.dQdz_poly(self.n_homo_coords, m, c) for (m,c) in zip(monomials, coefficients)]
        dQdz_monomials, dQdz_coeffs = list(zip(*dQdz_info))

        kappa = np.zeros((len(deformation_idx),), self.cdtype)
        for r, d_idx in tqdm(enumerate(deformation_idx)):
            print(f'Evaluating {d_idx}')
            a, b, c = d_idx
            kappa[r] = self.compute_yukawa_rst_component_batched(data, dQdz_monomials, dQdz_coeffs, a, b, c)

        return kappa
    
    def kahler_potential(self, p: Float[Array, "... i"], dQdz_monomials: List[np.array], 
                         dQdz_coeffs: List[np.array]) -> Complex[Array, "..."]:
        r"""Computes Kähler potential for moduli space metric at point $t$ in moduli space.
        $$ \mathcal{K}(t, \overline{t}) = - \log \int_{X_t} \Omega_t \wedge \overline{\Omega}_t.~.$$

        Parameters
        ----------
        p : Float[Array, "... i"]  
            2 * `complex_dim` real coords at which `fun` is evaluated. Note batch indices.
        dQdz_monomials : List[np.array]
            List of monomials corresponding to polynomial Jacobian $dQ/dz$.
        dQdz_coeffs : List[np.array]
            List of coefficients corresponding to polynomial Jacobian $dQ/dz$.

        Returns
        -------
        K : Complex[Array, "..."]
            Kähler potential at given point in complex structure moduli space.
        """
        weights, *_ = vmap(self.integration_weights_fn, in_axes=(0,None,None))(
                p, dQdz_monomials, dQdz_coeffs)
        vol_Omega = jnp.mean(weights)
        return - jnp.log(vol_Omega)

    @partial(jit, static_argnums=(0,6))
    def kappa_complete(self, p: Float[Array, "... i"], dQdz_monomials: List[np.array], 
                       dQdz_coeffs: List[np.array], weights=None, pb=None, 
                       output_variance: bool = False) -> Complex[Array, "h_21 h_21 h_21"]:
        r"""Computes full set of $h^{2,1} \times h^{2,1} \times h^{2,1}$ Yukawa couplings (three-point function) for a CY threefold. 
        On a fibre $X_s$ of the deformation family, where $s$ parameterises the moduli space, the $(a,b,c)$ component is given by,

        $$\kappa_{abc} = \int_{X_s} \Omega \wedge \left.\frac{d \Omega}{ds^a ds^b ds^c} \right\vert_{s=0}.$$

        Parameters
        ----------
        p : Float[Array, "... i"]  
            2 * `complex_dim` real coords at which `fun` is evaluated. Note batch indices.
        dQdz_monomials : List[np.array]
            List of monomials corresponding to polynomial Jacobian $dQ/dz$.
        dQdz_coeffs : List[np.array]
            List of coefficients corresponding to polynomial Jacobian $dQ/dz$.

        Returns
        -------
        int_kappa_abc: Complex[Array, "h_21 h_21 h_21"]
            Yukawa couplings at given point in complex structure moduli space.
        Notes
        -----
        Owing to vectorisation, this is significantly more efficient than computing individual couplings separately.
        """
        print(f'Compiling {self.kappa_complete.__qualname__}')

        if weights is None:
            weights, pb, *_ = vmap(self.integration_weights_fn, in_axes=(0,None,None))(
                p, dQdz_monomials, dQdz_coeffs)

        ones_mask = jnp.logical_not(jnp.isclose(p, jax.lax.complex(1.,0.)))
        dQdz_homo = vmap(alg_geo.evaluate_dQdz, in_axes=(0,None,None))(p, dQdz_monomials, dQdz_coeffs)
        Omega = vmap(alg_geo._holomorphic_volume_form, in_axes=(0,0,None,None,None))(
            p, dQdz_homo, self.n_hyper, self.n_homo_coords, self.ambient)

        # get Lie derivative of Ω along all diffeomorphism directions [..., h_{21}, n_inhomo_coords, n_homo_coords]
        dzeta_dz, dzeta_dzbar = vmap(self.zeta_jacobian_complete, in_axes=(0,None,None))(p, dQdz_monomials, dQdz_coeffs)
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
        print(f'{self.kappa_complete.__qualname__}, {kappa_abc.dtype}')

        if output_variance is True:
            # S_int_kappa = math_utils.shifted_variance(kappa_integrand, int_kappa_abc)
            S_int_re_kappa = jnp.mean((jnp.real(kappa_integrand) - jnp.real(int_kappa_abc))**2, axis=0)
            S_int_im_kappa = jnp.mean((jnp.imag(kappa_integrand) - jnp.imag(int_kappa_abc))**2, axis=0)
            return int_kappa_abc, S_int_re_kappa, S_int_im_kappa

        return int_kappa_abc
    
    def compute_yukawas_complete_batched(self, data, dQdz_monomials, dQdz_coeffs, output_variance=False):
        
        n = 0
        kappa_dtype = np.float64
        kappa = jnp.zeros((self.h_21, self.h_21, self.h_21), kappa_dtype)
        S_re_kappa = jnp.zeros_like(kappa, dtype=np.float64)
        S_im_kappa = jnp.zeros_like(S_re_kappa)
        _data_gen = lambda: zip(*data)

        for t, data in enumerate(tqdm(_data_gen(), total=len(data[0]))):
            p, w, pb = data
            B = p.shape[0]

            if output_variance is True:
                _kappa, _S_re_kappa, _S_im_kappa = self.kappa_complete(p, dQdz_monomials, dQdz_coeffs, w, pb, output_variance=True)
                re_kappa, S_re_kappa = math_utils.online_update_array(jnp.real(kappa), jnp.real(_kappa), n, B, S_re_kappa, _S_re_kappa)
                im_kappa, S_im_kappa = math_utils.online_update_array(jnp.imag(kappa), jnp.imag(_kappa), n, B, S_im_kappa, _S_im_kappa)
                kappa = jax.lax.complex(re_kappa, im_kappa)
            else:
                _kappa = self.kappa_complete(p, dQdz_monomials, dQdz_coeffs, w, pb)
                kappa = math_utils.online_update_array(kappa, _kappa, n, B)
            n += B

        if output_variance is True: 
            return kappa, S_re_kappa / (n-1), S_im_kappa / (n-1)

        return kappa
