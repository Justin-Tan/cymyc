"""
Utilities for scanning over moduli space to generate points, 
pullbacks and weights for fibres.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, jacfwd, vmap, random

import numpy as np
import time, argparse

from tqdm import tqdm
from functools import partial

# custom
from .. import alg_geo, dataloading, fubini_study
from ..utils import math_utils, pointgen_cicy

def gen_points(key, psi, n_p, coeff_fn, cy_dim, 
               monomials, ambient):
    
    # Define the defining polynomial
    coefficients = coeff_fn(psi)

    # Initialize point generator
    pg = pointgen_cicy.PointGenerator(key, cy_dim, monomials, coefficients, ambient)
    p = pg.sample_intersect_cicy(key, n_p)

    return p

def psi_1D_gen(args, key, psis, coeff_fn, psi_out):

    key, _key = random.split(key)
    np.save(os.path.join(args.out_path, psi_out), psis)
    print(f'Moduli space grid:, {psis.shape}')

    with tqdm(psis) as t:
        for i, psi in enumerate(t):
            t.set_description(f'ψ = {psi:.7f}')
            
            key, _key = random.split(key)
            p = gen_points(_key, psi, args.n_points, coeff_fn, args.cy_dim, 
                           args.monomials, args.ambient)

            np.savez_compressed(os.path.join(args.out_path, f'dataset_psi_{i}'),
                x=p, psi=psi)
    return

def psi_2D_gen(args, key, psis, coeff_fn, psi_out):

    key, _key = random.split(key)
    np.save(os.path.join(args.out_path, psi_out), psis)
    print(f'Moduli space grid:, {psis.shape}')

    for i in tqdm(range(psis.shape[0])):
        for j in tqdm(range(psis.shape[1]), leave=False):
            print(f'ψ = {psis[i,j]:.7f}')
            key, _key = random.split(key)
            p = gen_points(_key, psis[i,j], args.n_points, coeff_fn, args.cy_dim, 
                           args.monomials, args.ambient)

            np.savez_compressed(os.path.join(args.out_path, f'dataset_psi_{i}_{j}'),
                x=p, psi=psis[i,j])
    return

def aux_data_gen(files, poly_specification, coeff_fn, override=True, kappa_out=True):

    monomials, cy_dim, kmoduli, ambient = poly_specification()
    if not isinstance(monomials, list): monomials = [monomials]
    n_hyper, n_coords = len(monomials), monomials[0].shape[-1]
    conf_mat, p_conf_mat = math_utils._configuration_matrix(monomials, ambient) 
    t_degrees = math_utils._find_degrees(ambient, n_hyper, conf_mat)
    kmoduli_ambient = math_utils._kahler_moduli_ambient_factors(cy_dim, ambient, t_degrees)

    if n_hyper == 1:
        get_metadata = partial(alg_geo.compute_integration_weights, cy_dim=cy_dim)
        det_g_FS_fn = fubini_study.det_fubini_study_pb
    else:
        get_metadata = partial(alg_geo._integration_weights_cicy, n_hyper=n_hyper, cy_dim=cy_dim, 
                               n_coords=n_coords, ambient=ambient, kmoduli_ambient=kmoduli_ambient,
                               cdtype=np.complex128)
        det_g_FS_fn = partial(fubini_study.det_fubini_study_pb_cicy, n_coords=n_coords, ambient=tuple(ambient),
                cdtype=np.complex128)
    get_metadata = jit(get_metadata)

    B = 65536
    # override = False
    with tqdm(files) as t:
        for f in t:
            print(f)
            X = np.load(f)
            if 'w' in list(X.keys()) and (not override): continue
            p, psi = X['x'], X['psi']
            n = p.shape[0]
            t.set_description(f'ψ = {psi:.7f}')
            print(p.shape)

            coefficients = coeff_fn(psi)

            if n_hyper == 1:
                dQdz_info = alg_geo.dQdz_poly(n_coords, monomials[0], coefficients)
                dQdz_monomials, dQdz_coeffs = dQdz_info
            else:
                dQdz_info = [alg_geo.dQdz_poly(n_coords, m, c) for (m,c) in zip(monomials, coefficients)]
                dQdz_monomials, dQdz_coeffs = list(zip(*dQdz_info))

            data_batched = dataloading._online_batch(p, n, B, precision='low')
            weights, pullbacks, dVol_Omegas = [], [], []
            vol_Omega, vol_g = 0., 0.

            for data in tqdm(data_batched):
                _p = data
                w, pb, dVol_Omega, *_ = vmap(get_metadata, in_axes=(0,None,None))(_p, dQdz_monomials, dQdz_coeffs)
                weights.append(w)
                pullbacks.append(pb)
                dVol_Omegas.append(dVol_Omega)

                if kappa_out is True:
                    # compute Monge-Ampere proportionality constant
                    _det_g_FS_pb = vmap(det_g_FS_fn)(math_utils.to_real(_p), pb)
                    _vol_g = jnp.mean(w * _det_g_FS_pb / dVol_Omega)
                    vol_g = math_utils.online_update(vol_g, _vol_g, n, B)

                    _vol_Omega = jnp.mean(w)
                    vol_Omega = math_utils.online_update(vol_Omega, _vol_Omega, n, B)

            weights, pullbacks = np.squeeze(np.concatenate(weights, axis=0)), np.squeeze(np.concatenate(pullbacks, axis=0))
            dVol_Omegas = np.squeeze(np.concatenate(dVol_Omegas, axis=0))

            if kappa_out is True:
                kappa = vol_g / vol_Omega
                print(f'kappa: {kappa:.7f}')

            np.savez_compressed(f, x=p, psi=psi, w=weights, pb=pullbacks, dVol_Omega=dVol_Omegas)
            # np.savez_compressed(f, w=w, pb=pb, **X)

    return


def wp_scan_1D(args, wp, psis, deformation, coeff_fn):

    wp_mat = np.zeros_like(psis).astype(np.complex128)
    yukawa_mat = np.zeros_like(wp_mat)
    vol_Omega = np.zeros_like(wp_mat)
    args.out_path_yukawa = os.path.join(os.path.dirname(args.out_path), 'yukawas')
    args.out_path_vol_Omega = os.path.join(os.path.dirname(args.out_path), 'vol_Omega')

    with tqdm(psis) as psi_it:
        for i, psi in enumerate(psi_it):
            psi_it.set_description(f'ψ = {psi:.7f}')

            coefficients = coeff_fn(psi)

            # load points on fibre
            dpath = os.path.join(args.moduli_data_path, f'dataset_psi_{i}.npz')
            data_batched, _psi = dataloading._batch(dpath, args.B, x_train_key='x', metadata_key='psi')
            assert psi == _psi, f'Loaded data ({_psi}) does not match correct moduli pt! ({psi})'

            g_wp_i, vol_Omega_i = wp.compute_wp_batched_diagonal(data_batched, args.monomials, coefficients, deformation)
            kappa_i = wp.compute_yukawas_batched(data_batched, args.monomials, coefficients,
                        deformation, deformation, deformation)
            wp_mat[i], yukawa_mat[i], vol_Omega[i] = jnp.squeeze(g_wp_i), jnp.squeeze(kappa_i),\
                  jnp.squeeze(vol_Omega_i)
            print(f'\n g_wp(ψ) = {g_wp_i:.7f}')
            print(f'\n κ(ψ) = {kappa_i:.7f}')

            if i % 128 == 0 and i > 1:
                np.save(args.out_path, wp_mat)
                np.save(args.out_path_yukawa, yukawa_mat)
                np.save(args.out_path_vol_Omega, vol_Omega)

    np.save(args.out_path, wp_mat)
    np.save(args.out_path_yukawa, yukawa_mat)
    np.save(args.out_path_vol_Omega, vol_Omega)

    return wp_mat

def wp_scan_2D(args, wp, psis, deformation, coeff_fn):

    wp_mat = np.zeros_like(psis).astype(np.complex128)
    yukawa_mat = np.zeros_like(wp_mat)
    vol_Omega = np.zeros_like(wp_mat)
    args.out_path_yukawa = os.path.join(os.path.dirname(args.out_path), 'yukawas')
    args.out_path_vol_Omega = os.path.join(os.path.dirname(args.out_path), 'vol_Omega')

    for i in tqdm(range(psis.shape[0])):
        for j in tqdm(range(psis.shape[1]), leave=False):
            print(f"\n𝛙 = {psis[i,j]:.7f}\n")

            coefficients = coeff_fn(psis[i,j])

            # load points on fibre
            dpath = os.path.join(args.moduli_data_path, f'dataset_psi_{i}_{j}.npz')
            data_batched, _psi = dataloading._batch(dpath, args.B, x_train_key='x', metadata_key='psi')
            assert _psi == psis[i,j], f'Loaded data ({psis[i,j]}) does not match correct moduli pt! ({_psi})'

            # compute metric at point in moduli space along ONE deformation direction
            g_wp_ij, vol_Omega_ij = wp.compute_wp_batched_diagonal(data_batched, args.monomials, 
                coefficients, deformation)
            kappa_ij = wp.compute_yukawas_batched(data_batched, args.monomials, coefficients,
                        deformation, deformation, deformation)

            wp_mat[i,j] = jnp.squeeze(g_wp_ij)
            yukawa_mat[i,j], vol_Omega[i,j] = jnp.squeeze(kappa_ij), jnp.squeeze(vol_Omega_ij)
            print(f'\n g_wp(ψ) = {wp_mat[i,j]:.7f}')
            print(f'\n κ(ψ) = {kappa_ij:.7f}')

        np.save(args.out_path, wp_mat)
        np.save(args.out_path_yukawa, yukawa_mat)
        np.save(args.out_path_vol_Omega, vol_Omega)

    np.save(args.out_path, wp_mat)
    np.save(args.out_path_yukawa, yukawa_mat)
    np.save(args.out_path_vol_Omega, vol_Omega)

    return wp_mat

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def dictify(cmd_args, config):
    _dictify = lambda x: dict((n, getattr(x, n)) for n in dir(x) if not (n.startswith('__') or 'logger' in n))
    args_d, cmd_args_d = _dictify(config), vars(cmd_args)
    args_d.update(cmd_args_d)
    config = Struct(**args_d)
    return config

