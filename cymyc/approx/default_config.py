#!/usr/bin/env python3
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../..')))
import numpy as np

from functools import partial

from examples import poly_spec

"""
Default arguments. Entries may be manually overriden via
command line arguments.
"""

class directories(object):
    experiments = 'experiments'
    checkpoints = 'checkpoints'
    figures = 'figures'
    data = 'data'

class config(object):
    cy_dim = 3
    n_ambient_coords = -1
    n_epochs = 128
    batch_size = 1024
    eval_batch_size = 1024
    learning_rate = 1e-4
    n_steps = int(1e5)
    n_units = [48,48,48,48]
    log_interval = int(1e3)
    save_interval = 25  # epochs b/w model checkpoints 
    eval_interval = 8   # epochs b/w model eval. (default)
    eval_interval_t = int(4e3)  # iterations b/w model eval. (toggle via command line)
    gpu = 0
    cdtype = np.complex64
    periodic_eval = False

    n_units_harmonic = [64, 64, 128, 64, 42]

    # These need to correspond to the manifold you wish to examine,
    # see `examples/poly_spec` for conventions
    poly_specification = poly_spec.X33_spec
    coeff_fn = poly_spec.X33_coefficients
    deformation_fn = partial(poly_spec.X33_deformation, precision=cdtype)

    name = 'default'
    dataset = 'default'

    data_path = os.path.join(dataset, 'dataset.npz')
    metadata_path = os.path.join(dataset, 'metadata.pkl')
