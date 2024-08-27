# Disables memory preallocation on shared machines !!
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import jax
import jax.numpy as jnp
import numpy as np

from jax import random
from jax import jit, vmap

import optax
import argparse, time, os

from tqdm import tqdm
from functools import partial
from collections import defaultdict

# Custom imports
from . import losses
from ..utils import gen_utils as utils
cpu_device = jax.devices('cpu')[0]

def create_train_state(rng, model, optimizer, data_dim):
    rng, init_rng = random.split(rng)
    params = model.init(rng, jnp.ones([1, data_dim]))['params']
    opt_state = optimizer.init(params)
    return params, opt_state, init_rng

@partial(jit, static_argnums=(3,4,5))
def train_step(data, params, opt_state, metric_fn, optimizer, kappa):
    loss, dloss_dw = jax.value_and_grad(losses.objective_function, argnums=1)(data, params, metric_fn, kappa)
    param_updates, opt_state = optimizer.update(dloss_dw, opt_state, params)
    params = optax.apply_updates(params, param_updates)
    return params, opt_state, loss

def callback(epoch, t0, t, test_data, params, metric_fn, g_FS_fn, 
             config, storage, logger, mode='TRAIN'):

    loss_breakdown_dict = losses.loss_breakdown(test_data, params, metric_fn, g_FS_fn, config.kappa)
    loss_breakdown_dict = jax.device_get(loss_breakdown_dict)
    summary = jax.tree_util.tree_map(lambda x: x.item(), loss_breakdown_dict)

    logs = [f"{k}: {v:.4f}" for (k,v) in summary.items()]        
    logger.info(f"[{time.time()-t0:.1f}s]: [{mode}] | Epoch: {epoch} | Iter: {t}" + ''.join([f" | {log}" for log in logs]))

    [storage[k].append(v) for (k,v) in summary.items()]
    if (epoch % config.save_interval == 0) and (epoch > 0):  # in epochs
        utils.save_logs(storage, config.name, epoch)

    return storage


if __name__ == '__main__':

    from .default_config import config
    description = "CY metric learning."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-bs", "--batch_size", type=int, default=config.batch_size, help="Batch size.")
    parser.add_argument("-ds", "--dataset", type=str, default=config.dataset, help="Path to dataset.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=config.learning_rate, help="Learning rate.")
    parser.add_argument("-n_epochs", "--n_epochs", type=int, default=config.n_epochs, help="Number of training epochs.")
    parser.add_argument("-n_units", "--n_units", nargs='+', type=int, default=config.n_units, help="Number of hidden units in encoder/decoder.")
    parser.add_argument("-et", "--eval_interval", type=int, default=config.eval_interval, help="Evaluation frequency, in epochs.")
    parser.add_argument("-name", "--name", type=str, default='cy_exp', help="Experiment name for logs.")
    parser.add_argument("-ckpt", "--checkpoint", type=str, default=None, help="Path to checkpointed params.")
    cmd_args = parser.parse_args()

    from . import models
    from .. import dataloading

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
            config.eval_batch_size, logger=logger)
    dataset_size = A_train[0].shape[0]

    # initialize model
    if config.n_hyper > 1:
        model_class = models.LearnedVector_spectral_nn_CICY
    else:
        model_class = models.LearnedVector_spectral_nn

    model = model_class(config.n_ambient_coords, config.ambient, config.n_units)
    
    optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(config.learning_rate))
    # optimizer = optax.adamw(config.learning_rate)

    g_FS_fn, g_correction_fn, *_ = models.helper_fns(config)
    params, opt_state, init_rng = create_train_state(init_rng, model, optimizer, data_dim=config.n_ambient_coords * 2)
    # partial closure
    metric_fn = partial(models.ddbar_phi_model, g_ref_fn=g_FS_fn, g_correction_fn=g_correction_fn)

    t0 = time.time()
    logger.info(f'Using device(s), {jax.devices()}')
    logger.info(f'KAPPA: {config.kappa}')
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logger.info(f'Params (Count: {param_count})=========>>>')
    logger.info(jax.tree_util.tree_map(lambda x: x.shape, params))
    logger.info(model.tabulate(init_rng, jnp.ones([1, config.n_ambient_coords * 2])))

    for epoch in range(int(config.n_epochs)):

        if (config.periodic_eval is False) and (epoch % config.eval_interval == 0):
            val_data = dataloading.get_validation_data(val_loader, config.eval_batch_size, A_val, np_rng)
            storage = callback(epoch, t0, 0, val_data, params, metric_fn, g_FS_fn, config, storage, logger, mode='VAL')

        if epoch > 0: 
            train_loader = dataloading.data_loader(A_train, config.batch_size, np_rng)

        wrapped_train_loader = tqdm(train_loader, desc=f'Epoch {epoch}', total=dataset_size//config.batch_size, 
                                    colour='green', mininterval=0.1)

        for t, data in enumerate(wrapped_train_loader):

            params, opt_state, loss = train_step(data, params, opt_state, metric_fn, optimizer, config.kappa)
            
            if config.periodic_eval is True:
                if t % config.eval_interval_t == 0:
                    val_data = dataloading.get_validation_data(val_loader, config.eval_batch_size, A_val, np_rng)
                    storage = callback(epoch, t0, 0, val_data, params, metric_fn, g_FS_fn, config, storage, logger, mode='VAL')

            wrapped_train_loader.set_postfix_str(f"loss: {loss:.5f}", refresh=False)

        if epoch % config.save_interval == 0:
            utils.basic_ckpt(params, opt_state, config.name, f'{epoch}')

    utils.basic_ckpt(params, opt_state, config.name, 'FIN')
    utils.save_logs(storage, config.name, 'FIN')

