import numpy as np

import os, pickle
import itertools as it
from functools import partial
from typing import Iterable
import datetime
import logging

class directories(object):
    experiments = 'experiments'
    checkpoints = 'checkpoints'
    figures = 'figures'
    data = 'data'

EPS = 1e-6

_dictify = lambda x: dict((n, getattr(x, n)) for n in dir(x) if not (n.startswith('__') or 'logger' in n))

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def override_default_args(cmd_args, config):
    # Override default arguments from config file with provided command line arguments

    if cmd_args.dataset is not None:
        cmd_args.data_path = os.path.join(cmd_args.dataset, 'dataset.npz')
        cmd_args.metadata_path = os.path.join(cmd_args.dataset, 'metadata.pkl')

    args_d, cmd_args_d = _dictify(config), vars(cmd_args)
    args_d.update(cmd_args_d)
    config = Struct(**args_d)

    return config

def save_config(config):

    name = config.name
    os.makedirs(os.path.join(directories.experiments, name), exist_ok=True)
    cpath = os.path.join(directories.experiments, name, f"{name}_METADATA.pkl")
    print(f"Saving config file to {cpath}")
    config_d = _dictify(config)

    with open(cpath, 'wb') as handle:
        pickle.dump(config_d, handle)


def rpartial(func, *args, **kwargs):
    """Partially applies last arguments. 
       New keyworded arguments extend and override kwargs."""
       
    rfunc = lambda *a, **kw: func(*(a + args), **dict(kwargs, **kw))
    return rfunc


def generate_monomials(n, deg):
    r"""Yields a generator of monomials with degree deg in n variables.
    Args:
        n (int): number of variables
        deg (int): degree of monomials
    Yields:
        generator: monomial term
    """
    if n == 1:
        yield (deg,)
    else:
        for i in range(deg + 1):
            for j in generate_monomials(n - 1, deg - i):
                yield (i,) + j

def _patch_transitions(n_hyper, n_projective, degrees):
    n_transitions = 0
    for t in generate_monomials(n_projective, n_hyper):
        tmp_deg = [d-t[j] for j, d in enumerate(degrees)]
        n = np.prod(tmp_deg)
        if n > n_transitions:
            n_transitions = n

    return int(n_transitions)

def _generate_proj_indices(degrees):
    flat_list = []
    for i, p in enumerate(degrees):
        for _ in range(p):
            flat_list += [i]
    
    return np.array(flat_list, dtype=np.int32)

def _generate_all_patches(n_coords, n_transitions, degrees):
    r"""We generate all possible patches for CICYs. Note for CICYs with
    more than one hypersurface patches are generated on spot.
    """
    fixed_patches = []
    for i in range(n_coords):
        all_patches = np.array(
            list(it.product(*[[j for j in range(sum(degrees[:k]), sum(degrees[:k+1])) if j != i] \
                              for k in range(len(degrees))], repeat=1)))
        if len(all_patches) == n_transitions:
            fixed_patches += [all_patches]
        else:
            # need to padd if there are less than nTransitions.
            all_patches = np.tile(all_patches, (int(n_transitions/len(all_patches)) + 1, 1))
            fixed_patches += [all_patches[0:n_transitions]]
    fixed_patches = np.array(fixed_patches)
    return fixed_patches

def logger_setup(name, filepath, package_files=[]):

    logpath = os.path.join(directories.experiments, 
            '{}_{:%Y_%m_%d_%H:%M}.log'.format(name, datetime.datetime.now()))

    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s', 
                                  "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel('INFO'.upper())

    stream = logging.StreamHandler()
    stream.setLevel('INFO'.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    info_file_handler = logging.FileHandler(logpath, mode="a")
    info_file_handler.setLevel('INFO'.upper())
    info_file_handler.setFormatter(formatter)
    logger.addHandler(info_file_handler)

    logger.info(filepath)
    logger.propagate = False

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())
    return logger

def save_logs(storage, name, epoch):
    with open(os.path.join(directories.experiments, name, '{}_epoch_{}_{:%Y_%m_%d_%H:%M}_LOG.pkl'.format(name, epoch, datetime.datetime.now())), 'wb') as handle:
        pickle.dump(storage, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_params(params, name):
    from flax import serialization

    os.makedirs(os.path.join(directories.experiments, name), exist_ok=True)
    params_out = serialization.to_state_dict(params)
    with open(os.path.join(directories.experiments, name, '{}_{:%Y_%m_%d_%H:%M}_PARAMS.pkl'.format(name, datetime.datetime.now())), 'wb') as handle:
        pickle.dump(params_out, handle, protocol=pickle.HIGHEST_PROTOCOL)

def basic_ckpt(params, opt_state, name, epoch):
    from flax import serialization

    os.makedirs(os.path.join(directories.experiments, name), exist_ok=True)
    params_out = serialization.to_state_dict(params)
    opt_state_out = serialization.to_state_dict(opt_state)

    with open(os.path.join(directories.experiments, name, '{}_epoch_{}_{:%Y_%m_%d_%H:%M}_PARAMS.pkl'.format(name, epoch, datetime.datetime.now())), 'wb') as handle:
        pickle.dump(params_out, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(directories.experiments, name, '{}_epoch_{}_{:%Y_%m_%d_%H:%M}_OPT_STATE.pkl'.format(name,
        epoch, datetime.datetime.now())), 'wb') as handle:
        pickle.dump(opt_state_out, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_ckpt(init_params, init_opt_state, params_path, opt_state_path):
    from flax import serialization

    # Note content of initial states is not used except for shape/dtype assertion checking.
    with open(params_path, "rb") as handle:
        params_dict = pickle.load(handle)
    
    with open(opt_state_path, "rb") as handle:
        opt_state_dict = pickle.load(handle)
    
    params = serialization.from_state_dict(init_params, params_dict)
    opt_state = serialization.from_state_dict(init_opt_state, opt_state_dict)

    return params, opt_state

    
def load_params(init_params, params_path):
    from jax.tree_util import tree_map
    from flax import serialization

    # Note content of initial states is not used except for shape/dtype assertion checking.
    with open(params_path, "rb") as handle:
        params_dict = pickle.load(handle)

    # should probably match dtypes as well
    init_params_shapes = tree_map(np.shape, init_params)
    loaded_params_shapes = tree_map(np.shape, params_dict)
    n_units = frozenset([_v for v in init_params_shapes.values() for _v in v.values()])
    n_units_ckpt = frozenset([_v for v in loaded_params_shapes.values() for _v in v.values()])

    """
    n_units = [init_params[k]['kernel'].shape[-1] for k in sorted(init_params.keys()) if 'kernel' in
            init_params[k].keys()][:-1]
    n_units_ckpt = [params_dict[k]['kernel'].shape[-1] for k in sorted(params_dict.keys()) if 'kernel' in
            params_dict[k].keys()][:-1]
    """
    # n_units_ckpt = [params_dict[k]['kernel'].shape[-1] for k in params_dict.keys()][:-1]
    assert n_units == n_units_ckpt, f'Configured ({n_units}) and loaded ({n_units_ckpt}) models do not match.'

    params = serialization.from_state_dict(init_params, params_dict)
    return params


def save_metadata(poly_data, coefficients, kappa, dpath, data_out='dataset.npz', metadata_out='metadata.pkl',
                  topological_data=None):
    from . import math_utils
    from .. import alg_geo
    import sympy as sp

    _d = dict(zip(('monomials', 'cy_dim', 'kmoduli', 'ambient'), poly_data))
    config = Struct(**_d)
    config.coefficients = coefficients
    if not isinstance(config.monomials, list): 
        monomials = [config.monomials]
    else:
        monomials = config.monomials
    config.n_hyper, config.n_coords = len(monomials), monomials[0].shape[-1]
    config.n_ambient_coords = config.n_coords
    
    conf_mat, p_conf_mat = math_utils._configuration_matrix(monomials, config.ambient) 
    t_degrees = math_utils._find_degrees(config.ambient, config.n_hyper, conf_mat)
    config.kmoduli_ambient = math_utils._kahler_moduli_ambient_factors(config.cy_dim, config.ambient, t_degrees)

    if (config.n_hyper == 1) and (len(config.ambient) == 1):
        dQdz_info = alg_geo.dQdz_poly(config.n_coords, config.monomials, coefficients)
        config.dQdz_monomials, config.dQdz_coeffs = dQdz_info
    else:
        dQdz_info = [alg_geo.dQdz_poly(config.n_coords, m, c) for (m,c) in zip(monomials, coefficients)]
        config.dQdz_monomials, config.dQdz_coeffs = list(zip(*dQdz_info))

    data_path = os.path.abspath(os.path.join(dpath, data_out))
    meta_path = os.path.abspath(os.path.join(dpath, metadata_out))

    config.kappa = kappa
    print(f'kappa: {kappa:.7f}')

    if topological_data is not None:
        config.chi = topological_data['chi']
        config.vol = topological_data['vol']
        config.c2_w_J = topological_data['c2_w_J']

        vol = sp.sympify(config.vol)
        ts = " ".join([f"t_{i}" for i in range(len(config.kmoduli))])
        ts = np.array(sp.symbols(ts))

        # config.canonical_vol = topological_data['canonical_vol']
        if len(config.kmoduli) == 1:
            canonical_vol = float(vol.subs(ts, config.kmoduli[0]))
        else:
            canonical_vol = float(vol.subs(list(zip(ts, config.kmoduli))))
        config.canonical_vol = canonical_vol

    _dictify = lambda x: dict((n, getattr(x, n)) for n in dir(x) if not (n.startswith('__') or 'logger' in n))
    config_d = _dictify(config)

    with open(meta_path, 'wb') as handle:
        pickle.dump(config_d, handle)

    return config

def read_metadata(config, kmoduli=None, save=True):
    
    import sympy as sp

    metapath = os.path.join(config.metadata_path)
    with open(metapath, 'rb') as f:
        dataset_metadata = pickle.load(f)

    d = _dictify(config)
    d.update(dataset_metadata)
    config = Struct(**d)

    if (config.n_hyper == 1) and (len(config.ambient) == 1):
        config.dQdz_monomials = config.dQdz_monomials.astype(config.cdtype)
        config.dQdz_coeffs = config.dQdz_coeffs.astype(config.cdtype)
    else:
        config.dQdz_monomials = [m.astype(config.cdtype) for m in config.dQdz_monomials]
        config.dQdz_coeffs = [c.astype(config.cdtype) for c in config.dQdz_coeffs]

    if kmoduli is not None:
        config.kmoduli = kmoduli
        vol = sp.sympify(config.vol)
        ts = " ".join([f"t_{i}" for i in range(len(config.kmoduli))])
        ts = np.array(sp.symbols(ts))
        if len(config.kmoduli) == 1:
            canonical_vol = float(vol.subs(ts, config.kmoduli[0]))
        else:
            canonical_vol = float(vol.subs(list(zip(ts, config.kmoduli))))
        config.canonical_vol = canonical_vol

    if save is True: save_config(config)
    return config   

def random_params(rng, model, data_dim):
    from jax import random
    import jax.numpy as jnp
    
    rng, init_rng = random.split(rng)
    init_params = model.init(rng, jnp.ones([1, data_dim]))['params']
    return init_params, init_rng

def log_arrays(x):
    try:
        return x.item()
    except (ValueError, TypeError):
        return np.asarray(x)

def round_str(k,v, decimals=5):
    try:
        return f"{k}: {v:.5f}"
    except (ValueError, TypeError):
        return f"{k}: {np.round(v, decimals)}"
