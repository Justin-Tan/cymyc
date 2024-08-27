"""Simple in-memory dataloading. For datasets which don't fit in memory (why do you even have this?), 
consider https://github.com/google/grain. 
"""

from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np

def _online_batch(data, N, batch_size, aux=None, precision='high'):
    n_chunks = N // batch_size
    if precision != 'high':
        data = data.astype(np.complex64)
    _p = jnp.array_split(data, n_chunks)
    if aux is not None:
        _aux = [jnp.array_split(a, n_chunks) for a in aux]
        return (_p, *_aux)
    return _p

def _chunk(a: np.ndarray, n: int):
    return np.array_split(a, n)

def _batch(data_path, batch_size, x_train_key='x', metadata_key=None, logger=None,
           precision='high'):
    """In-memory batching in native `numpy`
    """

    data = np.load(data_path)
    p, meta = data[x_train_key], data[metadata_key]
    w, pb = data['w'], data['pb']
    if precision != 'high':
        p, w, pb = p.astype(np.complex64), w.astype(np.float32), pb.astype(np.complex64)
    N = p.shape[0]
    n_chunks = N // batch_size

    _p, _w, _pb = _chunk(p, n_chunks), _chunk(w, n_chunks), _chunk(pb, n_chunks)

    if logger is not None:
        logger.info(f'Dataset size: {p.shape}, meta: {meta:.7f}')
    else:
        print(f'Dataset size: {p.shape}, meta: {meta:.7f}')

    return (_p, _w, _pb), meta

def _batch_aux(data_path, batch_size, x_train_key='x', metadata_key=None, 
        aux_keys=None, logger=None, precision=np.float64):
    """In-memory batching in native `numpy` with unspecified auxillary data
    """

    data = np.load(data_path)
    p, meta = data[x_train_key], data[metadata_key]

    if precision != np.float64:
        p = p.astype(precision)

    if logger is not None:
        logger.info(f'Dataset size: {p.shape}, meta: {meta:.7f}')
    else:
        print(f'Dataset size: {p.shape}, meta: {meta:.7f}')

    N = p.shape[0]
    n_chunks = N // batch_size
    _p = _chunk(p, n_chunks)
    
    if aux_keys is not None:
        if precision is not np.float64:
            aux = [data[k].astype(precision) for k in aux_keys] 
        else:
            aux = [data[k] for k in aux_keys]
        _aux = [_chunk(a, n_chunks) for a in aux]
        return (_p, *_aux), meta

    return _p, meta


def data_loader(arrays, batch_size, np_rng):
    """Simple loading for in-memory data
    """
    N = arrays[0].shape[0]
    assert all(A.shape[0] == N for A in arrays)
    perm = np_rng.permutation(N)  # shuffle

    start, end = 0, batch_size
    while end <= N:
        batch_perm = perm[start:end]
        yield tuple(array[batch_perm] for array in arrays)
        start, end = end, end + batch_size


def initialize_loaders_train(np_rng, data_path, batch_size, x_train_key='x_train', x_val_key='x_val', 
        logger=None, precision=np.float32):
    """Dataloading based on the conventions of `utils/pointgen_cicy`
    """
    data = np.load(data_path)
    p_train, p_val = data[x_train_key].astype(precision), data[x_val_key].astype(precision)
    y_train, y_val = data['y_train'].astype(precision), data['y_val'].astype(precision)

    ds = data_loader((p_train, y_train[:,0], y_train[:,1]), batch_size, np_rng)
    ds_val = data_loader((p_val, y_val[:,0], y_val[:,1]), batch_size, np_rng)

    kappa = data['kappa']
    psi = data['psi']
    if logger is not None:
        logger.info(f'Dataset size: {p_train.shape}, kappa: {kappa:.7f}, psi: {psi:.7f}')
        logger.info(f"Vol[g]: {data['vol_g']:.7f}, Vol[Ω]: {data['vol_Omega']:.7f}")

    else:
        print(f'Dataset size: {p_train.shape}, kappa: {kappa:.7f}')
        print(f"Vol[g]: {data['vol_g']:.7f}, Vol[Ω]: {data['vol_Omega']:.7f}")

    return (p_train, y_train[:,0], y_train[:,1]), (p_val, y_val[:,0], y_val[:,1]), ds, ds_val, psi

def get_validation_data(val_loader, batch_size, A_val, np_rng):
    try:
        val_data = next(val_loader)
    except StopIteration:
        val_loader = data_loader(A_val, batch_size, np_rng)
        val_data = next(val_loader)
    return val_data


def shuffle_ds(ds, seed):
    ds = ds.shuffle(seed=seed)
    ds = ds.flatten_indices()
    return ds
