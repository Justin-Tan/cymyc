"""
Various internal functions for algebro-geometric computations. Computation of pullback routines based off
https://github.com/pythoncymetric/cymetric/blob/main/cymetric/models/fubinistudy.py.
"""

import jax
import numpy as np  # original CPU-backed NumPy
import jax.numpy as jnp

from jax import grad, jit, jacfwd, vmap

from functools import partial

import math
import sympy as sp

# custom
from .utils import math_utils
from . import fubini_study


def sym_poly(n_coords, monomials, coefficients):
    """
    Symbolic representation of polynomial given
    defining monomials and coefficients
    """
    x = sp.symbols(f'x:{n_coords}')
    poly = np.sum(coefficients * np.multiply.reduce(np.power(x, monomials), axis=-1))
    return poly

def evaluate_poly(points, monomials, coefficients):
    r"""
    Evaluates polynomial defined by monomials and coefficients
    Poly(x) = \sum_i coeff_i * monomial_i(x)
    """
    mono_eval = jnp.prod(jnp.power(points, monomials), axis=-1)
    return jnp.sum(coefficients * mono_eval, axis=-1)

def evaluate_poly_onp(x, monomials, coefficients):
    """
    Per-example evaluation of polynomial given defining monomials and coefficients.
    """
    poly = np.sum(coefficients * np.prod(np.power(x, monomials), axis=-1))
    return poly

def evaluate_poly_batch(x, monomials, coefficients):
    """
    Batch eval of polynomial given defining monomials and coefficients.
    """
    poly = np.sum(coefficients * np.prod(np.power(np.expand_dims(x,1), monomials), axis=-1), axis=-1)
    return poly

def dQdz_poly(n_coords, monomials, coefficients):
    
    dQdz_monomials = []
    dQdz_coeffs = []
    for i, m in enumerate(np.eye(n_coords, dtype=np.int32)):
        basis = monomials - m
        factors = coefficients * monomials[:, i]
        valid = np.ones(basis.shape[0], dtype=bool)
        valid[np.where(basis < 0)[0]] = False
        dQdz_monomials += [basis[valid]]
        dQdz_coeffs += [factors[valid]]
    basis_shapes = np.array([np.shape(mb) for mb in dQdz_monomials])
    m = len(basis_shapes)
    _dQdz_monomials = np.zeros((m, np.max(basis_shapes[:, 0]), m), dtype=np.int32)
    _dQdz_coeffs = np.zeros((m, np.max(basis_shapes[:, 0])), dtype=np.complex128)

    for i, m in enumerate(zip(dQdz_monomials, dQdz_coeffs)):
        _dQdz_monomials[i, :basis_shapes[i, 0]] += m[0]
        _dQdz_coeffs[i, :basis_shapes[i, 0]] += m[1]
    
    return _dQdz_monomials, _dQdz_coeffs

def evaluate_dQdz_batch(points, dQdz_monomials, dQdz_coefficients):
    """
    Evaluate Jacobian of defining polynomial w.r.t. local coordinates.
    No `vmap` for this one, TODO: test against `vmap` using points[:, jnp.newaxis, jnp.newaxis].
    """
    points = jnp.expand_dims(jnp.expand_dims(points,1),1)
    mono_eval = jnp.prod(jnp.power(points, dQdz_monomials), axis=-1)
    poly_eval = jnp.sum(dQdz_coefficients * mono_eval, axis=-1)
    
    return poly_eval

def evaluate_dQdz(p, dQdz_monomials, dQdz_coeffs):
    """
    Returns Jacobian of defining polynomial(s),
    dQdz, shape [n_hyper, n_coords] 
    """
    dQdz_all = jnp.stack([evaluate_poly(p, dm, dc) for (dm, dc) in        
                zip(dQdz_monomials, dQdz_coeffs)], axis=0)
    return dQdz_all

def poincare_residue(dQdz, elim_idx):
    r"""
    Return coefficients of the holomorphic (n,0)-form $\Omega$ on a CY n-fold 
    on patch U_i in affine coordinates.
    """
    return 1./dQdz[elim_idx]

def argmax_dQdz(points, dQdz):
    # Finds $$\argmax_i |dQ/dz_i|$$
    ones_mask = jnp.logical_not(jnp.isclose(points, jax.lax.complex(1.,0.)))
    elim_idx = jnp.argmax(jnp.abs(dQdz * ones_mask), axis=-1)
    return elim_idx

def argmax_dQdz_cicy(p, dQdz, n_hyper, n_coords, aux=False):
    """
    Finds suitable coordinates to eliminate for a submanifold embedded in a product of
    projective spaces by iteratively searching for \argmax_i \vert z_i \vert and eliminating.
    Also removes coords rescaled to unity. 
    """
    inv_ones_mask = jnp.isclose(p, jax.lax.complex(1.,0.))
    total_mask = jnp.logical_not(inv_ones_mask)

    elim_idx = jnp.zeros(n_hyper, dtype=np.int32)
    for i in range(n_hyper):
        elim_idx_i = jnp.argmax(jnp.abs(dQdz[i] * total_mask), axis=-1).astype(np.int32)
        elim_idx = elim_idx.at[i].set(elim_idx_i)
        elim_mask_i = (jnp.arange(n_coords) == elim_idx_i[np.newaxis])
        total_mask *= ~elim_mask_i

    if aux is True:
        return elim_idx, total_mask

    return elim_idx

def _create_pullback_mask_batch(points, dQdz_monomials, dQdz_coefficients):
    """
    Points is complex. Creates mask for pullbacks by finding $\argmax_i |dQ/dz_i|$.
    """
    dQdz = evaluate_dQdz_batch(points, dQdz_monomials, dQdz_coefficients)

    inv_ones_mask = jnp.isclose(points, jax.lax.complex(1.,0.))
    ones_idx = jnp.argmax(inv_ones_mask, axis=-1)
    ones_mask = jnp.logical_not(inv_ones_mask)
    
    elim_idx = jnp.argmax(jnp.abs(dQdz * ones_mask), axis=-1)
    elim_mask = jnp.repeat(jnp.expand_dims(jnp.arange(points.shape[-1]),0), \
                           points.shape[0], axis=0) == elim_idx[:,jnp.newaxis]

    dQdz_rescale = dQdz / jnp.expand_dims(dQdz[jnp.arange(len(dQdz)), elim_idx], axis=-1)

    # mask to eliminate k-th coordinate in patch U_k in projective space, and
    # j* coord for j* = \argmax_i |dQ/dz_i|
    good_coords_mask = jnp.logical_not(jnp.logical_or(inv_ones_mask, elim_mask))
    good_dQdz_rescale = dQdz_rescale[good_coords_mask].reshape(dQdz.shape[0], dQdz.shape[-1]-2)
    
    return elim_idx, ones_idx, dQdz, good_dQdz_rescale, good_coords_mask

def _create_pullback_mask(points, dQdz_monomials, dQdz_coefficients):
    """
    Points is complex. Creates mask for pullbacks by finding $\argmax_i |dQ/dz_i|$.
    Only works for hypersurfaces.
    """
    dQdz = evaluate_poly(points, dQdz_monomials, dQdz_coefficients)

    inv_ones_mask = jnp.isclose(points, jax.lax.complex(1.,0.))
    ones_idx = jnp.argmax(inv_ones_mask, axis=-1)
    ones_mask = jnp.logical_not(inv_ones_mask)
    
    elim_idx = jnp.argmax(jnp.abs(dQdz * ones_mask), axis=-1)
    elim_mask = jnp.arange(points.shape[-1]) == elim_idx[jnp.newaxis]

    dQdz_rescale = dQdz / jnp.expand_dims(dQdz[elim_idx], axis=-1)

    # mask to eliminate k-th coordinate in patch U_k in projective space, and
    # j* coord for j* = \argmax_i |dQ/dz_i|
    good_coords_mask = jnp.logical_not(jnp.logical_or(inv_ones_mask, elim_mask))
    good_dQdz_rescale = dQdz_rescale[jnp.nonzero(dQdz_rescale * good_coords_mask, size=dQdz.shape[-1]-2)]
    
    return elim_idx, ones_idx, dQdz, good_dQdz_rescale, good_coords_mask

def _patch_transition_function(points, patch_idx):
    """
    Implement transition functions between patches in projective space.
    Z_k = [z0/zk : z1/zk : ... : 1 (k) : ... : zm/zk (m) : ... : zn/zk]
    Z_m = [z0/zm : z1/zm : ... : zk/zm (k) : ... : 1 (m) : ... : zn/zm] = 1/(zm/zk) Z_k
    then generalize depending on projective space factors?
    """
    transition_factor = points[patch_idx]
    
    return points / transition_factor

def _transition_loss_setup(points, dQdz_info, all_patches, n_projective, n_transitions):

    points = math_utils.to_complex(points)
    dQdz_monomials, dQdz_coefficients = dQdz_info
    dQdz = evaluate_poly(points, dQdz_monomials, dQdz_coefficients)
    inv_ones_mask = jnp.isclose(points, jax.lax.complex(1.,0.))
    ones_mask = jnp.logical_not(inv_ones_mask)
    elim_idx = jnp.argmax(jnp.abs(dQdz * ones_mask), axis=-1)

    # possible patches other than the index fixed by the hypersurface polynomial
    other_patches_idx = all_patches[elim_idx].reshape(-1, n_projective).squeeze()    
    # cast each point in all other possible patches
    points_repeated = jnp.repeat(points[jnp.newaxis,:], n_transitions, axis=0)
    other_patch_points = vmap(_patch_transition_function)(points_repeated, other_patches_idx)
    other_patch_points_real = math_utils.to_real(other_patch_points)
    
    return other_patch_points_real

def _pullbacks(points, elim_idx, ones_idx, good_dQdz_rescale, good_coords_mask, cy_dim,
        cdtype=np.complex128):
    
    r"""
    Calculate Jacobian of inclusion map i: X -> CP^n
    Let $z^a$ denote local coords in CP^n, $x^b$ denote local coords on CY X, then
    outputs Jacobian $$\partial z^a/\partial x^b$$.
    """

    proj_dim = points.shape[-1]  # n_coords in projective space
    dzdx = jnp.zeros((proj_dim, cy_dim), dtype=cdtype)

    row_idx, col_idx = jnp.nonzero(good_coords_mask, size=cy_dim)[0], jnp.arange(cy_dim)
    dzdx = dzdx.at[row_idx, col_idx].set(jnp.ones(cy_dim, dtype=cdtype))
    
    dzdx = dzdx.at[elim_idx, ...].set(-good_dQdz_rescale)
    dzdx = dzdx.at[ones_idx, ...].set(jnp.zeros(cy_dim, dtype=cdtype))
    
    return jnp.transpose(dzdx, axes=(1,0))


# @partial(vmap, in_axes=(0,None,None,None,None,None))
@partial(jit, static_argnums=(3,4,5,6,7))
def _pullbacks_cicy(p, dQdz_monomials, dQdz_coeffs, n_hyper, cy_dim, n_coords,
                    aux, cdtype=np.complex64):    
    r"""
    Calculate Jacobian of inclusion map into product of projective spaces, 
    $ \iota: X -> \mathbb{P}^{n_1} x \mathbb{P}^{n_2} x ... x \mathbb{P}^{n_N}$ 
    Let $z^a$ denote local (homogeneous) coords in $\mathbb{P}^{n_j}$, $x^b$ denote local 
    coords on CY X, then outputs Jacobian $$\partial x^{a_j}/\partial z^b$$ of 
    shape [`cy_dim`, `n_coords`].

    Uses the implicit function theorem to calculate the columns of the
    Jacobian for variables eliminated by the defining polynomials. 
    """

    inv_ones_mask = jnp.isclose(p, jax.lax.complex(1.,0.))
    total_mask = jnp.logical_not(inv_ones_mask)  # gives good dims on CY where true
    
    dQdz = jnp.stack([evaluate_poly(p, dm, dc) for (dm, dc) in 
                zip(dQdz_monomials, dQdz_coeffs)], axis=0)

    elim_idx = jnp.zeros(n_hyper, dtype=np.int32)
    elim_mask = jnp.zeros(n_coords, dtype=np.int32)
                       
    # iterate to ensure two different defining polys don't eliminate
    # the same ambient space coordinate.
    for i in range(n_hyper):
        elim_idx_i = jnp.argmax(jnp.abs(dQdz[i] * total_mask), axis=-1).astype(np.int32)
        elim_idx = elim_idx.at[i].set(elim_idx_i)
        elim_mask_i = (jnp.arange(n_coords) == elim_idx_i)
        total_mask *= ~elim_mask_i
        elim_mask += elim_mask_i
    
    good_coords_idx = jnp.nonzero(total_mask, size=cy_dim)[0]

    # build Jacobians of defining polys w.r.t. CY and eliminated coords
    jac_elim = jnp.zeros((n_hyper, n_hyper), dtype=cdtype)
    jac_cy = jnp.zeros((n_hyper, cy_dim), dtype=cdtype)
    for i in range(n_hyper):
         jac_elim = jac_elim.at[i].set(dQdz[i][elim_idx])
         jac_cy = jac_cy.at[i].set(dQdz[i][good_coords_idx])

    # implicit function theorem
    dPdu = - jnp.linalg.solve(jac_elim, jac_cy)

    dzdx = jnp.zeros((cy_dim, n_coords), dtype=cdtype)
    cy_coord_idx = jnp.nonzero(total_mask, size=cy_dim)[-1]
    _ones = jnp.ones(cy_dim, dtype=cdtype)
    dzdx = dzdx.at[jnp.arange(cy_dim),cy_coord_idx].set(_ones)

    for i in range(n_hyper):
        dzdx = dzdx.at[..., elim_idx[i]].set(dPdu[i,:])

    if aux is True:
        jac_codim = project_to_codim(dQdz, elim_mask, codim=n_hyper)
        if n_hyper == 1: 
            Omega = jnp.squeeze(poincare_residue(jnp.squeeze(dQdz), elim_idx))
        else:
            Omega = 1./jnp.linalg.det(jac_codim)

        return jnp.squeeze(dzdx), jnp.squeeze(Omega)
    return jnp.squeeze(dzdx)

@partial(jit, static_argnums=(4,5,6,7,8))
def _pullbacks_cicy_set_dQ_elim(p, elim_mask, dQdz_monomials, dQdz_coeffs, n_hyper, cy_dim, n_coords,
                                aux, cdtype=np.complex64):
    """
    Like normal pb function except we specify the coordinates to be eliminated by the 
    polynomial constraints. Numerically unstable, for verification of transformation properties
    only! Remove for release?
    """

    inv_ones_mask = jnp.isclose(p, jax.lax.complex(1.,0.))
    total_mask = jnp.logical_not(inv_ones_mask)  # gives good dims on CY where true

    dQdz = jnp.stack([evaluate_poly(p, dm, dc) for (dm, dc) in
                zip(dQdz_monomials, dQdz_coeffs)], axis=0)

    total_mask *= ~elim_mask
    elim_idx = jnp.nonzero(elim_mask, size=n_hyper)[-1]
        
    good_coords_idx = jnp.nonzero(total_mask, size=cy_dim)[0]

    # build Jacobians of defining polys w.r.t. CY and eliminated coords
    jac_elim = jnp.zeros((n_hyper, n_hyper), dtype=cdtype)
    jac_cy = jnp.zeros((n_hyper, cy_dim), dtype=cdtype)
    for i in range(n_hyper):
         jac_elim = jac_elim.at[i].set(dQdz[i][elim_idx])
         jac_cy = jac_cy.at[i].set(dQdz[i][good_coords_idx])

    # implicit function theorem
    dPdu = - jnp.linalg.solve(jac_elim, jac_cy)

    dzdx = jnp.zeros((cy_dim, n_coords), dtype=cdtype)
    cy_coord_idx = jnp.nonzero(total_mask, size=cy_dim)[-1]
    _ones = jnp.ones(cy_dim, dtype=cdtype)
    dzdx = dzdx.at[jnp.arange(cy_dim),cy_coord_idx].set(_ones)

    for i in range(n_hyper):
        dzdx = dzdx.at[..., elim_idx[i]].set(dPdu[i,:])

    if aux is True:
        jac_codim = project_to_codim(dQdz, elim_mask, codim=n_hyper)
        if n_hyper == 1: 
            Omega = jnp.squeeze(poincare_residue(jnp.squeeze(dQdz), elim_idx))
        else:
            Omega = 1./jnp.linalg.det(jac_codim)
        return jnp.squeeze(dzdx), jnp.squeeze(Omega)
    return jnp.squeeze(dzdx)

def evaluate_dQdz(p, dQdz_monomials, dQdz_coeffs):
    # Returns Jacobian of defining polynomial(s),
    # dQdz, shape [n_hyper, n_coords] 
    dQdz_all = jnp.stack([evaluate_poly(p, dm, dc) for (dm, dc) in        
                zip(dQdz_monomials, dQdz_coeffs)], axis=0)
    return dQdz_all

def project_to_codim(A, mask, codim):
    return jnp.squeeze(A[:,jnp.nonzero(mask,size=codim)])

def _holomorphic_volume_form(p, dQdz, n_hyper, n_coords, ambient):
    # See reference https://arxiv.org/abs/hep-th/9411131
    inv_ones_mask = jnp.isclose(p, jax.lax.complex(1.,0.))
    total_mask = jnp.logical_not(inv_ones_mask)  # gives good dims on CY where true
    elim_idx = jnp.zeros(n_hyper, dtype=np.int32)
    elim_mask = jnp.zeros(n_coords, dtype=np.int32)
                       
    # iterate to ensure two different defining polys don't eliminate
    # the same ambient space coordinate.
    for i in range(n_hyper):
        elim_idx_i = jnp.argmax(jnp.abs(dQdz[i] * total_mask), axis=-1).astype(np.int32)
        elim_idx = elim_idx.at[i].set(elim_idx_i)
        elim_mask_i = (jnp.arange(n_coords) == elim_idx_i)
        total_mask *= ~elim_mask_i
        elim_mask += elim_mask_i

    if (n_hyper == 1) and (len(ambient) == 1): 
        return jnp.squeeze(poincare_residue(jnp.squeeze(dQdz), elim_idx))

    jac_codim = project_to_codim(dQdz, elim_mask, codim=n_hyper)
    Omega = 1./jnp.linalg.det(jac_codim)
    
    return Omega

def _integration_weights(points, dQdz, elim_idx, pullback_jac, cy_dim,
                                normalize_vol=False):
    r"""
    Compute weights for Monte Carlo integration. Points are
    sampled from some measure dA, hence the points must be reweighted as
    
    $$w_i = \frac{\Omega \wedge \bar{\Omega}}{dA}~,$$
    
    in order to numerically integrate w.r.t. the known Calabi-Yau 
    volume form :math: \Omega \wedge \bar{\Omega}.

    """

    # Calculate known CY volume form
    Omega = poincare_residue(dQdz, elim_idx)
    dVol_Omega = jnp.real(Omega * jnp.conjugate(Omega))

    # Calculate volume form of reference distribution
    real_points = math_utils.to_real(points)
    g_FS = fubini_study.fubini_study_metric_homo(real_points, normalization=jax.lax.complex(1.,0.))
    g_FS_pb = jnp.einsum('...ia,...ab,...jb->...ij', pullback_jac, g_FS, 
        jnp.conjugate(pullback_jac))
    
    # recall w ^ ... ^ w = n! det (g_{\mu \bar{\nu}}) dx^1 ^ dy^1 ^ ... dx^1 ^ dy^n
    dVol_ref = math.factorial(cy_dim) * jnp.real(jnp.linalg.det(g_FS_pb))

    weights = dVol_Omega / dVol_ref

    return weights, dVol_Omega, dVol_ref


def _fs_volume_form(p, pullbacks, dQdz_monomials, dQdz_coeffs, n_hyper, cy_dim, 
                    n_coords, ambient, kmoduli_ambient, normalize_vol=False, cdtype=np.complex128):
    p = math_utils.to_real(p)
    k_forms = []

    for t in kmoduli_ambient:
        g_FS_pb = fubini_study.fubini_study_metric_homo_pb_cicy(p, pullbacks, n_coords, 
            tuple(ambient), k_moduli=t, cdtype=cdtype)
        k_forms.append(g_FS_pb)

    eps_nd = jnp.array(math_utils.n_dim_eps_symbol(cy_dim))

    if cy_dim == 2:
        eps_contract = lambda *args: math_utils.eps_2D_contract(eps_nd, *args)
    elif cy_dim == 3:
        eps_contract = lambda *args: math_utils.eps_3D_contract(eps_nd, *args)
    elif cy_dim == 4:
        eps_contract = lambda *args: math_utils.eps_4D_contract(eps_nd, *args)
    else:
        raise ValueError('Only CY n-folds of dimensions < 4 currently supported.')

    dVol_ref = math.factorial(cy_dim) * eps_contract(*k_forms)
    return dVol_ref

# @partial(jit, static_argnums=(3,4,5))
def _integration_weights_cicy(p, dQdz_monomials, dQdz_coeffs, n_hyper, cy_dim, n_coords, ambient, 
        kmoduli_ambient, normalize_vol=False, cdtype=np.complex64):

    pullbacks, Omega = _pullbacks_cicy(p, dQdz_monomials, dQdz_coeffs, n_hyper, 
                                        cy_dim, n_coords, True, cdtype)

    dVol_ref =  _fs_volume_form(p, pullbacks, dQdz_monomials, 
            dQdz_coeffs, n_hyper, cy_dim, n_coords, tuple(ambient), kmoduli_ambient,
            cdtype)
    # recall w ^ ... ^ w = n! det (g_{\mu \bar{\nu}}) dx^1 ^ dy^1 ^ ... dx^1 ^ dy^n
    dVol_Omega = jnp.real(Omega * jnp.conjugate(Omega))
    weights = jnp.squeeze(jnp.real(dVol_Omega / dVol_ref))

    return weights, pullbacks, dVol_Omega, dVol_ref 

@partial(jit, static_argnums=(2,3))
def compute_pullbacks(points, dQdz_info, cy_dim, cdtype=np.complex128):
    dQdz_monomials, dQdz_coefficients = [di.astype(cdtype) for di in dQdz_info]
    elim_idx, ones_idx, dQdz, good_dQdz_rescale, good_coords_mask = _create_pullback_mask(points, dQdz_monomials, dQdz_coefficients)
    pbs = _pullbacks(points, elim_idx, ones_idx, good_dQdz_rescale, good_coords_mask, cy_dim, cdtype)

    return pbs

@partial(jit, static_argnums=(3,))
def compute_integration_weights(points, dQdz_monomials, dQdz_coefficients, cy_dim):
    elim_idx, ones_idx, dQdz, good_dQdz_rescale, good_coords_mask = _create_pullback_mask(points, dQdz_monomials, dQdz_coefficients)
    pbs = _pullbacks(points, elim_idx, ones_idx, good_dQdz_rescale, good_coords_mask, cy_dim)
    weights, dVol_omega, dVol_ref = _integration_weights(points, dQdz, elim_idx, pbs, cy_dim)
    
    return jnp.squeeze(weights), jnp.squeeze(pbs), jnp.squeeze(dVol_omega), jnp.squeeze(dVol_ref)


    
