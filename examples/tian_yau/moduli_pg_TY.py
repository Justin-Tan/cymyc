"""
Point generation utility for moduli space scans. Note this runs on CPU only.
"""

import os, multiprocessing
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count())
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import jax
from jax import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jax import jit, jacfwd, vmap, random

import numpy as np
import time, argparse


# custom
from cymyc.moduli import moduli_scan

from examples import poly_spec


def main(args):

    start = time.time()
    key = random.PRNGKey(int(start))
    os.makedirs(args.out_path, exist_ok=True)

    coeff_fn = poly_spec.tian_yau_KM_coefficients

    R_EPS = 0.5 * 1e-2
    THETA_EPS = 0.5 * 1e-2
    if args.grid is True:
        theta = np.linspace(0, 2.*np.pi - THETA_EPS, args.n_moduli)
        r = np.linspace(R_EPS, 2., args.n_moduli)
        r, theta = np.meshgrid(r, theta)
        psis = r*np.exp(1j*theta)
        psis[0,0] = 0.
        moduli_scan.psi_2D_gen(args, key, psis, coeff_fn, psi_out='psis_tian_yau_2D')
    else:
        psis = np.linspace(0., 2., args.n_moduli).astype(np.complex128)
        moduli_scan.psi_1D_gen(args, key, psis, coeff_fn, psi_out='psis_tian_yau_1D')

    delta_t = time.time() - start
    print(f'Time elapsed: {delta_t:.3f} s.')

    # pickle dictionary containing metadata here.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="pg-moduli-TY",
        description="Utility to generate points in fibre for grid on complex moduli space for TY.")
    parser.add_argument('-n_mod', '--n_moduli', type=int, help="Controls fineness of moduli space (r,θ) grid.", default=32)
    parser.add_argument('-n_CY', '--n_points', type=int, help="Number of points for MC integration on fibres.", default=100000)
    parser.add_argument('-o', '--out_path', type=str, help="Output directory holding results.", default="data/moduli_pts")
    parser.add_argument('-grid', '--grid', action='store_true', help='Toggle for generation over 2D grid.')
    args = parser.parse_args()

    _d = dict(zip(('monomials', 'cy_dim', 'kmoduli', 'ambient'), poly_spec.tian_yau_KM_spec()))
    args = moduli_scan.dictify(args, moduli_scan.Struct(**_d))
    main(args)
