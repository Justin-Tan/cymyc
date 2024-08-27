"""
Generates WP metric in complex moduli space for the TY manifold along a single x0*x1*x2 deformation.
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import vmap, random, jit

import numpy as np
import time, argparse

from tqdm import tqdm
from functools import partial

# custom
from cymyc import alg_geo, dataloading
from cymyc.moduli.wp import WP, WP_full
from cymyc.moduli import moduli_scan

from examples import poly_spec
from examples.tian_yau import TY_KM_poly_spec


def wp_scan_1D(args, wp, psis, coeff_fn, cdtype=np.complex128):

    n = psis.shape[0]
    vol_Omega = np.zeros_like(psis).astype(cdtype)
    vol_Omega_var = np.zeros_like(vol_Omega)

    g_wp_psi = []
    g_wp_psi_diag_var = []
    kappa_psi = []
    kappa_psi_stddev = []

    with tqdm(psis) as psi_it:
        for i, psi in enumerate(psi_it):
            psi_it.set_description(f'ψ = {psi:.7f}')

            coefficients = coeff_fn(psi)

            # load points on fibre
            dpath = os.path.join(args.moduli_data_path, f'dataset_psi_{i}.npz')
            data_batched, _psi = dataloading._batch(dpath, args.B, x_train_key='x', metadata_key='psi')
            assert psi == _psi, f'Loaded data ({_psi}) does not match correct moduli pt! ({psi})'

            dQdz_info = [alg_geo.dQdz_poly(wp.n_homo_coords, m, c) for (m,c) in zip(args.monomials, coefficients)]
            dQdz_monomials, dQdz_coeffs = list(zip(*dQdz_info))

            if args.variance_output is True:
                (_g_wp_psi, _g_wp_diag_var), (_vol_Omega_i, _vol_Omega_var) = \
                    wp.compute_wp_complete_batched(data_batched, dQdz_monomials, dQdz_coeffs, output_variance=True)
                _kappa_psi, _kappa_psi_re_stddev, _kappa_psi_im_stddev = wp.compute_yukawas_complete_batched(data_batched, dQdz_monomials, 
                        dQdz_coeffs, output_variance=True)
                g_wp_psi_diag_var.append(_g_wp_diag_var)
                kappa_psi_stddev.append(_kappa_psi_re_stddev)
                vol_Omega_var[i] = np.squeeze(_vol_Omega_var)
            else:
                _g_wp_psi, _vol_Omega_i = wp.compute_wp_complete_batched(data_batched, dQdz_monomials, dQdz_coeffs)
                _kappa_psi = wp.compute_yukawas_complete_batched(data_batched, dQdz_monomials, dQdz_coeffs)

            print('g_wp(ψ) shape', _g_wp_psi.shape)
            print('g_wp(ψ) dtype', _g_wp_psi.dtype)
            print('κ(ψ),shape', _kappa_psi.shape)
            print('κ(ψ),dtype', _kappa_psi.dtype)

            g_wp_psi.append(_g_wp_psi)
            kappa_psi.append(_kappa_psi)
            vol_Omega[i] = np.squeeze(_vol_Omega_i)

            np.save(args.out_path, g_wp_psi)
            np.save(args.out_path_yukawa, kappa_psi)
            np.save(args.out_path_g_wp_var, g_wp_psi_diag_var)

    g_wp_psi = np.stack(g_wp_psi, axis=0)
    kappa_psi = np.stack(kappa_psi, axis=0)

    np.save(args.out_path, g_wp_psi)
    np.save(args.out_path_vol_Omega, vol_Omega)
    np.save(args.out_path_yukawa, kappa_psi)

    if args.variance_output is True:
        g_wp_psi_diag_var = np.stack(g_wp_psi_diag_var, axis=0)
        kappa_psi_stddev = np.stack(kappa_psi_stddev, axis=0)
        np.save(args.out_path_g_wp_var, g_wp_psi_diag_var)
        np.save(args.out_path_yukawa_stddev, kappa_psi_stddev)
        np.save(args.out_path_vol_Omega_var, vol_Omega_var)

    return g_wp_psi, kappa_psi

def wp_scan_2D(args, wp, psis, deformations, coeff_fn):

    n = psis.shape[0]
    cs_dim = len(deformations)
    vol_Omega = np.zeros_like(psis).astype(np.complex64)
    g_wp_psi = np.zeros((n, n, cs_dim, cs_dim), dtype=np.complex64)
    kappa_psi = np.zeros((n, n, cs_dim, cs_dim, cs_dim), dtype=np.complex64)
    kappa_psi_stddev = np.zeros_like(kappa_psi)

    for i in tqdm(range(psis.shape[0])):
        for j in tqdm(range(psis.shape[1]), leave=False):
            print(f"\n𝛙 = {psis[i,j]:.7f}\n")

            coefficients = coeff_fn(psis[i,j])

            # load points on fibre
            dpath = os.path.join(args.moduli_data_path, f'dataset_psi_{i}_{j}.npz')
            data_batched, _psi = dataloading._batch(dpath, args.B, x_train_key='x', metadata_key='psi')
            assert _psi == psis[i,j], f'Loaded data ({psis[i,j]}) does not match correct moduli pt! ({_psi})'

            dQdz_info = [alg_geo.dQdz_poly(wp.n_homo_coords, m, c) for (m,c) in zip(args.monomials, coefficients)]
            dQdz_monomials, dQdz_coeffs = list(zip(*dQdz_info))

            _kappa_psi, _kappa_psi_stddev = wp.compute_yukawas_complete_batched(data_batched, dQdz_monomials, dQdz_coeffs)
            _g_wp_psi, _vol_Omega_psi = wp.compute_wp_complete_batched(data_batched, dQdz_monomials, dQdz_coeffs)

            print('g_wp(ψ), shape', _g_wp_psi.shape)
            print('κ(ψ), shape', _kappa_psi.shape)

            vol_Omega[i,j] = np.squeeze(_vol_Omega_psi)
            g_wp_psi[i,j] = jnp.squeeze(_g_wp_psi)
            kappa_psi[i,j] = jnp.squeeze(_kappa_psi)
            kappa_psi_stddev[i,j] = jnp.squeeze(_kappa_psi_stddev)

        np.save(args.out_path, g_wp_psi)
        np.save(args.out_path_yukawa, kappa_psi)
        np.save(args.out_path_yukawa_stddev, kappa_psi_stddev)
    
    np.save(args.out_path_vol_Omega, vol_Omega)
    np.save(args.out_path, g_wp_psi)
    np.save(args.out_path_yukawa, kappa_psi)
    np.save(args.out_path_yukawa_stddev, kappa_psi_stddev)

    return g_wp_psi, kappa_psi


def main(args, deformations):

    start = time.time()
    key = random.PRNGKey(int(start))
    psis = np.load(args.psi_data_path)
    print(f'# ψs:, {psis.shape}')    

    args.out_path = os.path.join(args.moduli_data_path, args.out_fname)
    args.out_path_g_wp_var = os.path.join(args.moduli_data_path, 'g_wp_diag_var')
    args.out_path_yukawa = os.path.join(os.path.dirname(args.out_path), 'kappas')
    args.out_path_yukawa_stddev = os.path.join(os.path.dirname(args.out_path), 'kappas_stddev')
    args.out_path_vol_Omega = os.path.join(os.path.dirname(args.out_path), 'vol_Omega')
    args.out_path_vol_Omega_var = os.path.join(os.path.dirname(args.out_path), 'vol_Omega_var')

    coeff_fn = poly_spec.tian_yau_KM_coefficients
    wp = WP_full(args.cy_dim, args.monomials, args.ambient, deformations)
    
    if args.grid is True:
        g_wp_psi, kappa_psi = wp_scan_2D(args, wp, psis, deformations, coeff_fn)
    else:
        g_wp_psi, kappa_psi = wp_scan_1D(args, wp, psis, coeff_fn)
    
    np.save(args.out_path, g_wp_psi)
    np.save(args.out_path_yukawa, kappa_psi)
    
    delta_t = time.time() - start
    print(f'Time elapsed: {delta_t:.3f} s.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="WP-TY",
        description="Weil-Petersson metric calculation for TY manifold (unidirectional).")
    parser.add_argument('-psis', '--psi_data_path', type=str, help="Path to array of moduli space pts.", required=True)
    parser.add_argument('-mod_data', '--moduli_data_path', type=str, help="Path to directory holding fibre points in CS moduli space.", 
                        default="data/moduli_pts/")
    parser.add_argument('-o', '--out_fname', type=str, help="Output file to store results.",default="wp_vals_TY_full_psi_1D.npy")
    parser.add_argument('-grid', '--grid', action='store_true', help='Toggle for scan over 2D grid.')
    parser.add_argument('-B', '--B', type=int, help='Batch size in running average. Adjust to tradeoff memory/speed.', default=16834)
    parser.add_argument('-vo', '--variance_output', action='store_true', help='Output MC variance in calculation of Yukawas.')
    args = parser.parse_args()

    _d = dict(zip(('monomials', 'cy_dim', 'kmoduli', 'ambient'), poly_spec.tian_yau_KM_spec()))
    args = moduli_scan.dictify(args, moduli_scan.Struct(**_d))
    args.n_hyper = len(args.monomials)

    # deformations = TY_KM_poly_spec.TY_KM_deformations()
    deformations = TY_KM_poly_spec.TY_KM_deformations_expanded()

    kappa_deformation_idx = poly_spec.tian_yau_KM_yukawas()
    kappa_deformation_quarks_idx = TY_KM_poly_spec.tian_yau_KM_yukawas_quarks()

    main(args, deformations)
