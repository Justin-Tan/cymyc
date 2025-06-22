"""Fast point sampling from hypersurfaces in projective space.
"""

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np  # original CPU-backed NumPy
import jax.numpy as jnp

from jax import jit, vmap, random

import time, os
from functools import partial
from collections import defaultdict

import sympy as sp

# custom
from . import math_utils
from . import gen_utils as utils
from .. import alg_geo, fubini_study

cpu_device = jax.devices('cpu')[0]

def S2np1_uniform(key, n_p, n, dtype=np.float64):
    """
    Sample `n_p` points uniformly on $S^{2n+1}$, treated as CP^n
    """
    # return random.uniform(key, (n,))*jnp.pi, random.uniform(key, (n,)) * 2 * jnp.pi
    x = random.normal(key, shape=(n_p, 2*(n+1)), dtype=dtype)
    x_norm = x / jnp.linalg.norm(x, axis=1, keepdims=True)
    sample = math_utils.to_complex(x_norm.reshape(-1, n+1, 2))

    return jnp.squeeze(sample)

def univariate_coefficient_data(cy_dim, monomials, coefficients):
    """
    Computes defining polynomial evaluated at line parameterized by `t`,
    `Q(p + t * q)`. Returns data (monomial exponents and coefficients) 
    required to compute coefficients of univariate polynomial in `t`.
    """
    c_dim = cy_dim + 2
    ps, qs, t = sp.symarray('p', c_dim), sp.symarray('q', c_dim), sp.symbols('t')
    Q = alg_geo.evaluate_poly_onp(ps + t * qs, monomials, coefficients)
    
    t_poly = sp.Poly(Q, t)
    t_coeffs_sym = defaultdict(int)
    generators = set()
    for (p,), t_coeff in t_poly.terms():
        t_coeffs_sym[p] = t_coeff
        generators = generators.union(set(t_coeff.as_poly().gens))
    
    generators = sorted(list(generators), key=str)
    t_coeffs_data = defaultdict(lambda: (jnp.array(()), jnp.array(())))
    for deg, poly_deg_t in t_coeffs_sym.items():
        deg_t_terms = sp.Poly(poly_deg_t, generators).terms()
        exponents = np.array([e for e, _ in deg_t_terms])
        coeffs = np.array([c for _, c in deg_t_terms], dtype=np.complex128)
        t_coeffs_data[deg] = (jnp.asarray(exponents), jnp.asarray(coeffs))

    return t_coeffs_data, generators

# @partial(jit, static_argnums=(3,))
def root_solver(p, q, t_coeffs_data, t_generators):

    pq = jnp.concatenate([p,q], axis=-1)
    assert pq.shape[-1] == len(t_generators)

    # compute coefficient for each degree t^n, descending
    t_coeffs = []
    for (monos_deg_t, coeffs_deg_t) in t_coeffs_data:
        t_coeffs.append(
            alg_geo.evaluate_poly(pq, monos_deg_t, coeffs_deg_t))
    
    t_coeffs = jnp.array(t_coeffs)

    # Bezout's theorem says this returns `c_dim` intersection points
    with jax.default_device(cpu_device):
        t_roots = jnp.roots(t_coeffs, strip_zeros=False)
    pts = p + jnp.expand_dims(t_roots, 1) * q
    return pts, t_coeffs
    
def sample_intersect_hypersurface(key: random.PRNGKey, n_p: int, 
                                  cy_dim: int, monomials: np.array, 
                                  coefficients: np.array, LOCUS_TOL: float = 1e-10):

    """Samples from manifold defined as a hypersurface in projective space
    by solving for the intersection 'Q(p + t * q)'.
    """

    _key, key = random.split(key, 2)

    # Generate points on S^{2n+1} (S^{2n+1}/U(1) \cong CP^n)
    c_dim = cy_dim + 2  # homo. coords plus hypersurface constraint
    n_intersect = np.ceil(n_p / c_dim).astype(int)
    sphere_pts = S2np1_uniform(_key, 2*n_intersect, cy_dim+1)
    p, q = jnp.split(sphere_pts, 2)

    # solve for intersection of line with hypersurface, compute 
    # Q(p + t * q), find coefficients of terms of each power symbolically
    t_coeffs_data, generators = univariate_coefficient_data(cy_dim, monomials, coefficients)
    
    # find coeffs and pass to root solver - TODO: extend to gpu.
    # on cpu because `linalg.eig` not supported on gpu.
    pts, t_coeffs = vmap(root_solver, in_axes=(0,0,None,None))(
        p, q, t_coeffs_data.values(), tuple(generators))
    pts = pts.reshape(-1, c_dim)
    abs_poly_val = jnp.abs(vmap(alg_geo.evaluate_poly, in_axes=(0,None,None))(pts, 
                    monomials, coefficients))

    # recall Bezout's theorem guarantees `c_dim` intersecting points
    # rescale points - return homogeneous coords with $\max{|z_i|} = 1$
    pts, *_ = math_utils.rescale(pts.reshape(-1, c_dim)[:n_p])

    abs_poly_val = jnp.abs(vmap(alg_geo.evaluate_poly, in_axes=(0,None,None))(pts, 
                    monomials, coefficients))
    pts = pts[abs_poly_val < LOCUS_TOL]

    return pts

if __name__ == "__main__":

    import argparse
    from examples import poly_spec

    parser = argparse.ArgumentParser(
        description='Hypersurface point generation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-o', '--output_path', type=str, help="Path to the output directory for points.", required=True)
    parser.add_argument('-n_p', '--num_pts', type=int, help="Number of points to generate.", default=100000)
    parser.add_argument('-val', '--val_frac', type=float, help="Percentage of points to use for validation.", default=0.2)
    parser.add_argument('-psi', '--psi', type=float, help="Complex moduli parameter.", default=0.0)
    args = parser.parse_args()

    start = time.time()
    key = random.PRNGKey(42) # int(start))
    n_devs, n_p = len(jax.devices('cpu')), args.num_pts
    v_p = int(args.val_frac * n_p)

    # Example polynomial specification
    # ========================
    # poly_specification = poly_spec.mirror_quintic_spec
    # coeff_fn = poly_spec.mirror_quintic_coefficients
    poly_specification = poly_spec.fermat_quartic_spec
    coeff_fn = poly_spec.fermat_quartic_coefficients
    psi = args.psi
    coefficients = coeff_fn(args.psi)
    # ========================

    monomials, cy_dim, kmoduli, ambient = poly_specification()
    print('poly coefficients', coefficients)

    n_coords, n_hyper = monomials.shape[-1], 1
    n_fold = np.sum(ambient) - n_hyper
    dQdz_info = alg_geo.dQdz_poly(n_coords, monomials, coefficients)
    dQdz_monomials, dQdz_coeffs = dQdz_info
    n_p = args.num_pts

    p = sample_intersect_hypersurface(key, n_p + v_p, cy_dim, monomials, coefficients)
    
    print(f'{p.shape[0]} points generated.')
    # check CY condition
    abs_poly_val = jnp.abs(vmap(alg_geo.evaluate_poly, in_axes=(0,None,None))(p, monomials, coefficients)).max()
    print(f'Max locus violation: {abs_poly_val:.7e}')
    
    det_g_FS_fn = fubini_study.det_fubini_study_pb

    weights, pullbacks, dVol_Omegas, *_ = vmap(alg_geo.compute_integration_weights, in_axes=(0,None,None,None))(
        p, dQdz_monomials, dQdz_coeffs, cy_dim)

    p = math_utils.to_real(p)

    _det_g_FS_pb = vmap(det_g_FS_fn)(p, pullbacks)
    vol_g = jnp.mean(weights * _det_g_FS_pb / dVol_Omegas).item()
    vol_Omega = jnp.mean(weights).item()

    kappa = (vol_g / vol_Omega)

    conf, p_conf = math_utils._configuration_matrix((monomials,), ambient)
    p_conf = np.array(p_conf)
    chi, c2_w_J, vol, canonical_vol = math_utils.Pi(p_conf, kmoduli, cy_dim)
    topological_data = {'chi': chi, 'c2_w_J': c2_w_J, 'vol': vol, 'canonical_vol': canonical_vol}
    print('Wall data', topological_data)
    print('Volume', vol)
    print('Volume at chosen Kahler moduli', canonical_vol)

    print(f'Saving under {args.output_path}/ ...')
    os.makedirs(args.output_path, exist_ok=True)
    f = os.path.join(args.output_path, 'dataset.npz')


    if args.val_frac == 0.:
        np.savez_compressed(f, x=p, w=weights, dVol_Omega=dVol_Omegas, 
                            kappa=kappa, vol_g=vol_g, vol_Omega=vol_Omega)
    else:
        p_train, p_val = np.array_split(p, (n_p,))
        w_train, w_val = np.array_split(weights, (n_p,))
        dVol_Omega_train, dVol_Omega_val = np.array_split(dVol_Omegas, (n_p,))
        # pb_train, pb_val = np.array_split(pullbacks, (n_p,))

        y_train = np.stack((w_train, dVol_Omega_train), axis=-1)
        y_val = np.stack((w_val, dVol_Omega_val), axis=-1)

        np.savez_compressed(f, x_train=p_train, y_train=y_train, x_val=p_val, 
                            y_val=y_val, # pb_train=pb_train, pb_val=pb_val, 
                            kappa=kappa, vol_g=vol_g, vol_Omega=vol_Omega,
                            psi=psi)
        
    metadata = utils.save_metadata(poly_specification(), coefficients, kappa, args.output_path,
                                   topological_data=topological_data)
    delta_t = time.time() - start
    print(f'Time elapsed: {delta_t:.3f} s')

