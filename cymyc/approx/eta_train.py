# Disables memory preallocation on shared machines !
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import jax
import jax.numpy as jnp
import numpy as np

from jax import random
from jax import jit, vmap

import optax
from flax import linen as nn

import argparse, time, os

from tqdm import tqdm
from functools import partial
from collections import defaultdict

# Custom imports
from cymyc import alg_geo
from cymyc.utils import gen_utils as utils

jax.config.update("jax_enable_x64", False)
cpu_device = jax.devices('cpu')[0]

def create_train_state(rng, model, optimizer, data_dim):
    rng, init_rng = random.split(rng)
    params = model.init(rng, jnp.ones([1, data_dim]))['params']
    opt_state = optimizer.init(params)
    return params, opt_state, init_rng

@partial(jit, static_argnums=(3,4))
def train_step(data, params, opt_state, objective_function, optimizer):
    print(f'Compiling {train_step.__qualname__}')
    
    loss, dloss_dw = jax.value_and_grad(objective_function, argnums=1)(data, params)
    param_updates, opt_state = optimizer.update(dloss_dw, opt_state, params)
    params = optax.apply_updates(params, param_updates)
    return params, opt_state, loss

def callback(loss_breakdown, epoch, t0, t, test_data, params, config, storage, logger, mode='TRAIN'):

    loss_breakdown_dict = loss_breakdown(test_data, params)
    loss_breakdown_dict = jax.device_get(loss_breakdown_dict)
    # summary = jax.tree_util.tree_map(lambda x: x.item(), loss_breakdown_dict)
    summary = jax.tree_util.tree_map(utils.log_arrays, loss_breakdown_dict)

    logs = [f"{utils.round_str(k,v)}" for (k,v) in summary.items()]
    logger.info(f"[{time.time()-t0:.1f}s]: [{mode}] | Iter: {t}" + ''.join([f" | {log}" for log in logs]))

    if storage is not None:
        [storage[k].append(v) for (k,v) in summary.items()]
        if epoch % config.save_interval == 0:  # in epochs
            utils.save_logs(storage, config.name, epoch)

    return storage


if __name__ == '__main__':

    from .default_config import config

    description = "Harmonic (0,1) bundle-valued form learning."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-bs", "--batch_size", type=int, default=config.batch_size, help="Batch size.")
    parser.add_argument("-ds", "--dataset", type=str, default=config.dataset, help="Path to dataset.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=config.learning_rate, help="Learning rate.")
    parser.add_argument("-n_epochs", "--n_epochs", type=int, default=config.n_epochs, help="Number of training epochs.")
    parser.add_argument("-n_units_harmonic", "--n_units_harmonic", nargs='+', type=int, default=config.n_units_harmonic,
    help="Number of hidden units in network for harmonic forms.")
    parser.add_argument("-et", "--eval_interval", type=int, default=config.eval_interval, help="Evaluation frequency, in iterations.")
    parser.add_argument("-name", "--name", type=str, default='harmonic_cy_exp', help="Experiment name for logs.")
    parser.add_argument("-ckpt", "--metric_checkpoint", type=str, required=True, help="Path to checkpointed params for metric NN.")
    cmd_args = parser.parse_args()

    from cymyc import dataloading
    from cymyc.approx import models, harmonic

    seed = int(time.time()) # 42
    rng = random.PRNGKey(seed)
    rng, init_rng = random.split(rng)

    # Override default arguments from config file with provided command line arguments
    config = utils.override_default_args(cmd_args, config)
    config = utils.read_metadata(config)  # load dataset metadata
    storage = defaultdict(list)
    logger = utils.logger_setup(config.name, filepath=os.path.abspath(__file__))

    # Load data
    np_rng = np.random.default_rng()
    logger.info(f'Loading data from {config.data_path}')
    A_train, A_val, train_loader, val_loader, psi = dataloading.initialize_loaders_train(np_rng, config.data_path, 
            config.batch_size, logger=logger)
    dataset_size = A_train[0].shape[0]

    # Set multiple deformations here, e.g. for Tian-Yau,
    from examples.tian_yau import TY_KM_poly_spec
    # defo_idx = [4, 9, 11, 12, 13, 17, 18, 21, 22]
    defo_idx = [9,18]
    deformations = TY_KM_poly_spec.TY_KM_deformations_expanded()
    config.deformations = [deformations[idx] for idx in defo_idx]
    # config.deformations = [config.deformation_fn]

    # initialize model
    if (config.n_hyper > 1) or (len(config.ambient) > 1):
        model_class = models.LearnedVector_spectral_nn_CICY
        eta_model_class = models.CoeffNetwork_spectral_nn_CICY
        eta_model = eta_model_class(config.n_ambient_coords, config.ambient, config.n_units_harmonic,
               activation=nn.gelu, h_21=len(config.deformations))
    else:
        model_class = models.LearnedVector_spectral_nn

    # load metric model
    g_model = model_class(config.n_ambient_coords, config.ambient, config.n_units)
    _params, init_rng = utils.random_params(init_rng, g_model, data_dim=config.n_ambient_coords * 2)
    g_params = utils.load_params(_params, config.metric_checkpoint)  # parameters for trained metric NN
    logger.info(g_model.tabulate(init_rng, jnp.ones([1, config.n_ambient_coords * 2])))

    # optimizer = optax.adam(config.learning_rate)
    optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(config.learning_rate))
    params, opt_state, init_rng = create_train_state(init_rng, eta_model, optimizer, data_dim=config.n_ambient_coords * 2)

    t0 = time.time()
    logger.info(f'Using device(s), {jax.devices()}')
    logger.info(f'κ: {config.kappa}')
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logger.info(f'Params (Count: {param_count})=========>>>')
    logger.info(jax.tree_util.tree_map(lambda x: x.shape, params))
    logger.info(eta_model.tabulate(init_rng, jnp.ones([1, config.n_ambient_coords * 2])))

    # Weil-Petersson stuff
    monomials, cy_dim, kmoduli, ambient = config.poly_specification()
    coefficients = config.coeff_fn(psi)

    if (config.n_hyper == 1) and (len(config.ambient) == 1):
        monomials = [monomials]
        coefficients = [coefficients]

    dQdz_info = [alg_geo.dQdz_poly(config.n_ambient_coords, m, c) for (m,c) in zip(monomials, coefficients)]
    dQdz_monomials, dQdz_coeffs = list(zip(*dQdz_info))

    # this could be cleaned up
    g_FS_fn, g_correction_fn, pb_fn = models.helper_fns(config)
    metric_fn = jax.tree_util.Partial(models.ddbar_phi_model, params=g_params, g_ref_fn=g_FS_fn, g_correction_fn=g_correction_fn)


    harmonic_wp = harmonic.HarmonicFull(config.cy_dim, monomials, config.ambient, config.deformations, dQdz_monomials,
                                    dQdz_coeffs, metric_fn, pb_fn, config.coeff_fn, psi)

    for epoch in range(int(config.n_epochs)):

        if config.periodic_eval is False:
            val_loader, val_data = dataloading.get_validation_data(val_loader, config.eval_batch_size, A_val, np_rng)
            storage = callback(harmonic_wp.loss_breakdown, epoch, t0, 0, val_data, params, config, storage, logger, mode='VAL')
        
        if epoch > 0: 
            train_loader = dataloading.data_loader(A_train, config.batch_size, np_rng)

        wrapped_train_loader = tqdm(train_loader, desc=f'Epoch {epoch}', total=dataset_size//config.batch_size, 
                                    colour='green', mininterval=0.1)

        for t, data in enumerate(wrapped_train_loader):
            params, opt_state, loss = train_step(data, params, opt_state, harmonic_wp.objective_function, optimizer) 

            if config.periodic_eval is True:
                if t % config.eval_interval == 0:
                    val_loader, val_data = dataloading.get_validation_data(val_loader, config.eval_batch_size, A_val, np_rng)
                    _ = callback(harmonic_wp.loss_breakdown, epoch, t0, 0, data, params, config, storage, logger, mode='TRAIN')
                    storage = callback(harmonic_wp.loss_breakdown, epoch, t0, 0, val_data, params, config, storage, logger, mode='VAL')
            wrapped_train_loader.set_postfix_str(f"loss: {loss:.5f}", refresh=False)

        if epoch % config.save_interval == 0:
            utils.basic_ckpt(params, opt_state, config.name, f'{epoch}')

    utils.basic_ckpt(params, opt_state, config.name, 'FIN')
    utils.save_logs(storage, config.name, 'FIN')

