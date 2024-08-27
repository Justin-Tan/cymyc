"""
Utility to generate pullbacks and weights for moduli space scans for TY. 
"""

import os, multiprocessing, glob
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import jax
from jax import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jax import jit, jacfwd, vmap, random, pmap

import numpy as np
import time, argparse

# custom
from cymyc import alg_geo
from cymyc.moduli import moduli_scan

from examples import poly_spec

def main(args):

    start = time.time()
    key = random.PRNGKey(int(start))
    files = glob.glob(os.path.join(args.input_path, '*.npz'))
    
    moduli_scan.aux_data_gen(files, poly_spec.tian_yau_KM_spec, coeff_fn=poly_spec.tian_yau_KM_coefficients)
    delta_t = time.time() - start
    print(f'Time elapsed: {delta_t:.3f} s.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="metadata-moduli-TY",
        description="Utility to generate weights/pbs in fibre for grid on complex moduli space for TY.")
    parser.add_argument('-i', '--input_path', type=str, help="Output directory holding points.", required=True)
    args = parser.parse_args()

    _d = dict(zip(('monomials', 'cy_dim', 'kmoduli', 'ambient'), poly_spec.tian_yau_KM_spec()))
    args = moduli_scan.dictify(args, moduli_scan.Struct(**_d))

    main(args)
