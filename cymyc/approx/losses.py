r"""Objective functions and diagnostics for approximation of metrics of vanishing Ricci curvature on Calabi-Yaus.
"""
import jax
import numpy as np  # original CPU-backed NumPy
import jax.numpy as jnp

from jax import jit, jacfwd, vmap

from functools import partial
from jaxtyping import Array, Float, Complex, ArrayLike
from typing import Callable, Mapping, Union, Tuple, Optional

# custom
from . import measures
from .. import alg_geo, chern_gauss_bonnet, curvature, fubini_study
from ..utils import math_utils

@partial(jit, static_argnums=(2,3))
def monge_ampere_loss(g_pred: Complex[Array, "cy_dim cy_dim"], dVol_Omega: Float[Array, ""], 
                      kappa: float = 1., norm_order: float = 1.) -> jnp.ndarray:
    r"""Computes the integrand of the Monge-Ampère loss,

    $$
    \mathcal{L}_{\textsf{MA}} := \int_X \left\Vert 1 - \frac{1}{\kappa} \frac{\det g}{\Omega \wedge \overline{\Omega}} \right\Vert^p d\mu_{\Omega}~.
    $$

    This enforces the condition that $\omega^n \propto \Omega \wedge \bar{\Omega}$ up to some constant $\kappa \in \mathbb{C}$,
    which is a consequence of Ricci-flatness. 
    Parameters
    ----------
    g_pred :  Complex[Array, "dim dim"]
        Predicted metric $g_{\mu \overline{\nu}}$ in local coordinates.
    dVol_Omega : Float[Array, ""]
        $\Omega \wedge \bar{\Omega}$ in local coordinates.
    kappa : float, optional
        Proportionality constant between the canonical volume form and volume form induced by `g_pred`.
    norm_order : float, optional
        Order of norm of loss, by default 1.

    Returns
    -------
    jnp.ndarray
        Computed Monge-Ampère loss.

    Notes
    -----
    The parameter $\kappa \in \mathbb{C}$ denotes the constant of proportionality between $\Omega \wedge \overline{\Omega}$ 
    and the volume form $\bigwedge^n \tilde{\omega}$ induced by the approximate Kähler form $\tilde{\omega}$. In
    general,
    $$ \bigwedge^n \omega = h(z) \, \Omega \wedge \overline{\Omega} ~,$$
    for some holomorphic function $h$, but $h$ is constant for the Ricci-flat Kaehler form. Supply this if this is known 
    beforehand, e.g. for an ansatz which remains cohomologous to some known known reference metric.
    """
    det_g = jnp.real(jnp.linalg.det(g_pred))
    r = det_g / dVol_Omega 

    return jnp.abs(1. - 1./kappa * r)**norm_order


def ricci_tensor_loss(p: Float[Array, "i"], metric_fn: Callable[[Array], Array], 
                      pullbacks: Complex[Array, "cy_dim i"] = None, ricci_scalar_out: bool = False, 
                      norm_order: float = None) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, 
                                                                            jnp.ndarray]]:
    r"""Computes the norm of the Ricci tensor, in local coordinates. Here we use the fact that
    the Ricci curvature on a Kähler manifold is computable as,
    $$
    \textsf{Ric} = \partial \overline{\partial} \log \det g_{\mu \bar{\nu}}~.
    $$

    The Ricci tensor loss is then given by $\int_X \left\Vert \textsf{Ric} \right\Vert^p d\mu_{\Omega}$.

    Parameters
    ----------
    p : array_like  
        2 * `complex_dim` real coords at which `fun` is evaluated. Shape [i].
    metric_fn : callable
        Function representing metric tensor in local coordinates $g : \mathbb{R}^m -> \mathbb{C}^{a,b...}$.
    pullbacks : array_like, optional
        Pullback matrices from ambient to projective variety. If supplied, computes
        Ricci curvature on variety.
    ricci_scalar_out : bool, optional
        Toggle to output Ricci scalar, default False.
    norm_order : Optional[float], optional
        Order of the norm, default None (corresponding to 2-norm).

    Returns
    -------
    Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]
        Computed Ricci tensor loss, and optionally Ricci scalar.

    See Also
    --------
    curvature.ricci_form_kahler, curvature.ricci_tensor_kahler.
    """
    print(f'Compiling {ricci_tensor_loss.__qualname__}')
    # Ricci form in ambient space
    ricci_form = curvature.ricci_form_kahler(p, metric_fn, pullbacks)
    ricci_tensor = 1.j * ricci_form

    if ricci_scalar_out is True:
        g_pred = metric_fn(p)
        g_inv = jnp.linalg.inv(g_pred)
        R = jnp.einsum('...ij, ...ji->...', g_inv, ricci_tensor)
        return ricci_tensor, R, g_pred
        # return jnp.linalg.norm(ricci_tensor, ord=norm_order)/np.prod(ricci_tensor.shape), R

    return jnp.linalg.norm(ricci_tensor, ord=norm_order)/np.prod(ricci_tensor.shape)

def kahler_loss(p: Float[Array, "i"], pullbacks: Complex[Array, "cy_dim i"], 
                metric_fn: Callable[[Array], Array], norm_order: float = 2) -> jnp.ndarray:
    r"""Computes the integrand of the condition arising from the closedness of the 
    Kähler form $\omega$,
    $$
    d\omega = 0 \implies g_{\mu \overline{\nu}, \rho} = g_{\rho \overline{\nu}, \mu}~,
    $$
    with a similar condition for the antiholomorphic derivative. See Nakahara (8.82), 
    page 331 for more details. Note this should be exactly zero 
    for the FS metric + {exact correction}!

    Parameters
    ----------
    p : array_like
        2 * `complex_dim` real coordinates at which `metric_fn` is evaluated. Shape [i].
    pullbacks : array_like
        Pullback matrices from ambient to projective variety.
    metric_fn : callable
        Function representing the metric tensor in local coordinates $g : \mathbb{R}^m -> \mathbb{C}^{a,b...}$.
    norm_order : float, optional
        Order of norm, by default 2.

    Returns
    -------
    jnp.ndarray
        Computed Kähler loss.
    """
    
    dim = p.shape[-1]//2  # complex dimension
    real_jac_fun = jacfwd(metric_fn)(p)
    real_jac_fun = jnp.squeeze(real_jac_fun)
    dg_dx = real_jac_fun[..., 0:dim]
    dg_dy = real_jac_fun[..., dim:]

    dg_dz = 0.5 * (dg_dx - 1.j * dg_dy)
    dg_dz_bar = 0.5 * (dg_dx + 1.j * dg_dy)
      
    # dz^i/dx^j - e.g. for quintic (3,5)
    dg_dz_pb = jnp.einsum('...ij,...abj->...abi', pullbacks, dg_dz)
    dg_dz_bar_pb = jnp.einsum('...ij,...abj->...abi', jnp.conjugate(pullbacks), dg_dz_bar)
    
    holo_diff = dg_dz_pb - jnp.einsum('...ijk->...kji', dg_dz_pb)
    antiholo_diff = dg_dz_bar_pb - jnp.einsum('...ijk->...ikj', dg_dz_bar_pb)

    return jnp.sum(jnp.abs(holo_diff)**norm_order + jnp.abs(antiholo_diff)**norm_order)


def volume_loss(data: Tuple[ArrayLike, ArrayLike, ArrayLike], g_FS_pb: jnp.ndarray, 
                g_pred: jnp.ndarray, norm_order: float = 1) -> jnp.ndarray:    
    r"""
    Computes the discrepancy between the volume computed using the respective 
    volume forms constructed from the Fubini-Study metric and predicted metric. As the 
    corresponding Kähler forms are cohomologous, this should be zero.

    $$
        \mathcal{L}_{\text{vol}} = \left\Vert \textsf{vol}_{\text{FS}} - \textsf{vol}_{\text{CY}} \right\Vert^p~, 
        \quad \textsf{vol}_g := \int_X d^nx \, \det g~.
    $$

    Parameters
    ----------
    data : Tuple[ArrayLike, ArrayLike, ArrayLike]
        Tuple containing input points, integration weights and canonical volume form
        $\Omega \wedge \bar{\Omega}$ in local coords.
    g_FS_pb : jnp.ndarray
        Pullback of the Fubini-Study metric.
    g_pred : jnp.ndarray
        Predicted metric in local coordinates.
    norm_order : float, optional
        Order of norm, default 1.

    Returns
    -------
    jnp.ndarray
        Computed volume loss.
    """
    p, weights, dVol_Omega = data

    # compute normalization of Fubini-Study volume implicit in the weights
    vol_FS = jnp.mean(weights/dVol_Omega * jnp.real(jnp.linalg.det(g_FS_pb)))
    # compute volume according to volume form constructed from the NN metric
    vol_CY = jnp.mean(weights/dVol_Omega * jnp.real(jnp.linalg.det(g_pred)))

    return jnp.abs(vol_FS - vol_CY)**norm_order


def kappa_estimate(det_g, weights, dVol_Omega):
    r = det_g / dVol_Omega 
    vol_Omega = jnp.mean(weights)
    vol_CY = jnp.mean(r * weights)
    kappa = vol_CY / vol_Omega
    return kappa


@partial(jit, static_argnums=(2,3))
def objective_function(data: Tuple[ArrayLike, ArrayLike, ArrayLike], 
                       params: Mapping[str, Array],
                       metric_fn: Callable[[ArrayLike], jnp.ndarray], 
                       kappa: Optional[float] = None) -> jnp.ndarray:
    r"""Default objective function for optimisation of Ricci-flat metrics, using only the
    Monge-Ampère loss.

    Parameters
    ----------
    data : Tuple[ArrayLike, ArrayLike, ArrayLike]
        Tuple containing input points, integration weights, and canonical volume form
        $\Omega \wedge \bar{\Omega}$ in local coordinates.
    params : Mapping[Str, Array]
        Model parameters stored as a dictionary - keys are the module names
        registered upon initialisation and values are the parameter values.
    metric_fn: Callable[[ArrayLike], jnp.ndarray], 
        Function representing metric tensor in local coordinates, 
        $g : \mathbb{R}^m -> \mathbb{C}^{a,b...}$.
    kappa : float, optional
        Proportionality constant between the canonical volume form and volume form
        induced by approximate metric.

    Returns
    -------
    jnp.ndarray
        Computed objective function value.
    """
    p, weights, dVol_Omega = data
    g_pred = vmap(metric_fn, in_axes=(0,None))(p, params)
    if kappa is None:
        kappa = jax.lax.stop_gradient(kappa_estimate(jnp.real(jnp.linalg.det(g_pred)), weights, dVol_Omega))

    ma_loss = vmap(monge_ampere_loss, in_axes=(0,0,None))(g_pred, dVol_Omega, kappa)
    loss = ma_loss

    return jnp.mean(loss * weights)

@partial(jit, static_argnums=(2,3))
def objective_function_ricci(data: Tuple[ArrayLike, ArrayLike, ArrayLike], 
                             params: Mapping[str, Array],
                             metric_fn: Callable[[ArrayLike], jnp.ndarray], 
                             pb_fn: Callable[[ArrayLike], jnp.ndarray]) -> jnp.ndarray:
    p, weights, _ = data
    metric_fn = jax.tree_util.Partial(metric_fn, params=params)
    pb = vmap(pb_fn)(math_utils.to_complex(p))
    ricci_tensor, R, g_pred = vmap(ricci_tensor_loss, in_axes=(0,None,0,None))(p, metric_fn, pb, True)
    loss = jnp.linalg.norm(jnp.abs(ricci_tensor - 0.5 * g_pred * jnp.expand_dims(R, axis=(-1,-2)))) / np.prod(ricci_tensor.shape[1:])

    return jnp.mean(loss * weights)  # probably should norm this.

def loss_breakdown(data, params, metric_fn, g_FS_fn, kappa=None,
        canonical_vol=None):
    p, weights, dVol_Omega = data

    # full closure for \del \bar{\del} operations
    metric_fn = jax.tree_util.Partial(metric_fn, params=params)
        
    g_FS_ambient, pullbacks = vmap(g_FS_fn)(p)
    g_FS_pb = jnp.einsum('...ia,...ab,...jb->...ij', pullbacks, g_FS_ambient, jnp.conjugate(pullbacks))

    g_pred = vmap(metric_fn)(p)
    cy_dim = g_pred.shape[-1]
    det_g_pred = jnp.real(jnp.linalg.det(g_pred))

    if kappa is None:
        kappa = jax.lax.stop_gradient(kappa_estimate(det_g_pred, weights, dVol_Omega))

    ma_loss = vmap(monge_ampere_loss, in_axes=(0,0,None))(g_pred, dVol_Omega, kappa)

    k_loss = vmap(kahler_loss, in_axes=(0,0,None))(p, pullbacks, metric_fn)
    v_loss = volume_loss(data, g_FS_pb, g_pred)
    vol_CY = jnp.mean(weights / dVol_Omega * det_g_pred) # prefactor is choice of convention
    vol_Omega = jnp.mean(weights)
    _sigma_measure = measures.sigma_measure(data, metric_fn)

    # Chunk Hessian computations - large batch sizes OOM on GPU
    n_chunks = 4
    data_chunked = jax.tree_util.tree_map(partial(lambda n, x: jnp.array_split(x, n), n_chunks), (*data, pullbacks))
    _p, _w, _dVol_Omega, _pb = data_chunked
    ricci_tensor, chi_form, n = [], 0., 0
    S_chi_form = 0.

    for i in range(n_chunks):
        B = _p[i].shape[0]
        _data = (_p[i], _w[i], _dVol_Omega[i])
        _ricci_tensor = -1.j * vmap(curvature.ricci_form_kahler, in_axes=(0,None,0))(_p[i], metric_fn, _pb[i])
        _ricci_measure = measures.ricci_measure(_data, _pb[i], metric_fn, cy_dim)
        
        # no Pfaffian
        _chi_form_integrand = vmap(chern_gauss_bonnet.euler_characteristic_form, in_axes=(0,0,None))(_data, _pb[i], metric_fn)
        _chi_form = jnp.mean(_chi_form_integrand)
        S_chi_form_i = jnp.mean(jnp.square(_chi_form_integrand - _chi_form), axis=0)

        chi_form, S_chi_form = math_utils.online_update_array(chi_form, _chi_form, n, B, S_chi_form, S_chi_form_i)
        ricci_tensor.append(_ricci_tensor)
        n += B

    chi_form = chi_form * canonical_vol / vol_CY
    g_inv = jnp.linalg.inv(g_pred)
    ricci_tensor = jnp.vstack(ricci_tensor)
    R = jnp.real(jnp.einsum('...ij, ...ji->...', g_inv, ricci_tensor))
    ricci_tensor_norm = jnp.mean(vmap(jnp.linalg.norm)(ricci_tensor)) / np.prod(ricci_tensor.shape[1:])
    einstein_tensor_norm = jnp.linalg.norm(jnp.abs(ricci_tensor - 0.5 * g_pred * jnp.expand_dims(R, axis=(-1,-2)))) \
        / np.prod(ricci_tensor.shape[1:])

    return {'einstein_norm': jnp.mean(einstein_tensor_norm * weights), 
            'monge_ampere_loss': jnp.mean(ma_loss * weights),
            'ricci_tensor_norm': ricci_tensor_norm, 'chi_form': chi_form,
            'kahler_loss': k_loss.mean(), 'ricci_scalar': R.mean(), 'det_g': det_g_pred.mean(),
            'vol_loss': v_loss.mean(), 'vol_CY': vol_CY, 'vol_Omega': vol_Omega,
            'sigma_measure': _sigma_measure, 'ricci_measure': _ricci_measure}

def ma_proportionality(p, weights, config):
    r"""
    Calculates proportionality constant between the rival volume forms $\Omega \wedge \bar{\Omega}$ and $\omega^n$. 
    """

    if (config.n_hyper == 1) and (len(config.ambient) == 1):
        get_metadata = partial(alg_geo.compute_integration_weights, config.dQdz_monomials, config.dQdz_coeffs, 
                cy_dim=config.cy_dim)
        g_FS_fn = fubini_study.fubini_study_metric_homo_pb_precompute
    else:
        get_metadata = partial(alg_geo._integration_weights_cicy, dQdz_monomials=config.dQdz_monomials,
                dQdz_coeffs=config.dQdz_coeffs, n_hyper=config.n_hyper, cy_dim=config.cy_dim, 
                n_coords=config.n_coords, ambient=config.ambient, kmoduli_ambient=config.kmoduli_ambient)
        g_FS_fn = partial(fubini_study.fubini_study_metric_homo_pb_cicy, n_coords=config.n_coords, ambient=config.ambient)

    weights, pullbacks, dVol_Omega, *_= vmap(get_metadata)(p)
    g_FS_pb = vmap(g_FS_fn)(p, pullbacks)

    vol_Omega = jnp.mean(jnp.squeeze(weights))
    det_g_FS_pb = jnp.real(jnp.squeeze(jnp.linalg.det(g_FS_pb))) 

    vol_g = jnp.mean(weights * det_g_FS_pb / dVol_Omega)
    kappa = vol_g / vol_Omega
    print(f'kappa: {kappa:.7f}')

    return kappa

partial(jit, static_argnums=(2,3,4))
def objective_function_perelman(data: Tuple[ArrayLike, ArrayLike, ArrayLike], 
                                joint_params: Mapping[str, Array],
                                metric_fn: Callable[[ArrayLike], jnp.ndarray], 
                                global_fn: Callable[[ArrayLike], jnp.ndarray], 
                                pb_fn: Callable[[ArrayLike], jnp.ndarray]) -> jnp.ndarray:
    p, weights, _ = data
    metric_fn = jax.tree_util.Partial(metric_fn, params=joint_params['g_model'])
    f_pred = vmap(global_fn, in_axes=(0,None))(p, joint_params['f_model'])

    pb = vmap(pb_fn)(math_utils.to_complex(p))
    ricci_tensor, R, g_pred = vmap(ricci_tensor_loss, in_axes=(0,None,0,None))(
        p, metric_fn, pb, True)
    g_inv = jnp.linalg.inv(g_pred)
    grad_f = vmap(curvature.del_z, in_axes=(0,None,None,None))(p, global_fn, True, joint_params['f_model'])
    grad_f = jnp.einsum('...ai,...i->...a', pb, grad_f)
    grad_f_sq = jnp.einsum('...ij,...i,...j->...', g_inv, jnp.conjugate(grad_f), grad_f)
    integrand = (R + grad_f_sq) * jnp.exp(-f_pred)
    log_integrand = jnp.log(R + grad_f_sq) - f_pred
    return jnp.mean(integrand * weights)

def loss_breakdown_perelman(data, joint_params, metric_fn, global_fn, g_FS_fn, pb_fn, kappa=None,
        canonical_vol=None):
    p, weights, dVol_Omega = data
    g_params, f_params = joint_params['g_params'], joint_params['f_params']

    # full closure for \del \bar{\del} operations
    metric_fn = jax.tree_util.Partial(metric_fn, params=g_params)

    g_FS_ambient, pullbacks = vmap(g_FS_fn)(p)
    g_FS_pb = jnp.einsum('...ia,...ab,...jb->...ij', pullbacks, g_FS_ambient, jnp.conjugate(pullbacks))

    g_pred = vmap(metric_fn)(p)
    cy_dim = g_pred.shape[-1]
    det_g_pred = jnp.real(jnp.linalg.det(g_pred))

    if kappa is None:
        kappa = jax.lax.stop_gradient(kappa_estimate(det_g_pred, weights, dVol_Omega))

    ma_loss = vmap(monge_ampere_loss, in_axes=(0,0,None))(g_pred, dVol_Omega, kappa)

    k_loss = vmap(kahler_loss, in_axes=(0,0,None))(p, pullbacks, metric_fn)
    v_loss = volume_loss(data, g_FS_pb, g_pred)
    vol_CY = jnp.mean(weights / dVol_Omega * det_g_pred) # prefactor is choice of convention
    vol_Omega = jnp.mean(weights)
    _sigma_measure = measures.sigma_measure(data, metric_fn)

    # Chunk Hessian computations - large batch sizes OOM on GPU
    n_chunks = 4
    data_chunked = jax.tree_util.tree_map(partial(lambda n, x: jnp.array_split(x, n), n_chunks), (*data, pullbacks))
    _p, _w, _dVol_Omega, _pb = data_chunked
    ricci_tensor, n = [], 0
    chi_form, S_chi_form = 0., 0.
    F_functional, S_F_functional = 0., 0.

    for i in range(n_chunks):
        B = _p[i].shape[0]
        _data = (_p[i], _w[i], _dVol_Omega[i])
        _ricci_tensor = -1.j * vmap(curvature.ricci_form_kahler, in_axes=(0,None,0))(_p[i], metric_fn, _pb[i])
        _ricci_measure = measures.ricci_measure(_data, _pb[i], metric_fn, cy_dim)

        # no Pfaffian
        _chi_form_integrand = vmap(chern_gauss_bonnet.euler_characteristic_form, in_axes=(0,0,None))(_data, _pb[i], metric_fn)
        _chi_form = jnp.mean(_chi_form_integrand)
        S_chi_form_i = jnp.mean(jnp.square(_chi_form_integrand - _chi_form), axis=0)

        _F_functional = objective_function_perelman(_data, joint_params, metric_fn, global_fn, pb_fn)

        chi_form, S_chi_form = math_utils.online_update_array(chi_form, _chi_form, n, B, S_chi_form, S_chi_form_i)
        F_functional = math_utils.online_update_array(F_functional, _F_functional, n, B)

        ricci_tensor.append(_ricci_tensor)
        n += B

    chi_form = chi_form * canonical_vol / vol_CY
    g_inv = jnp.linalg.inv(g_pred)
    ricci_tensor = jnp.vstack(ricci_tensor)
    R = jnp.real(jnp.einsum('...ij, ...ji->...', g_inv, ricci_tensor))
    ricci_tensor_norm = jnp.mean(vmap(jnp.linalg.norm)(ricci_tensor)) / np.prod(ricci_tensor.shape[1:])
    einstein_tensor_norm = jnp.linalg.norm(jnp.abs(ricci_tensor - 0.5 * g_pred * jnp.expand_dims(R, axis=(-1,-2)))) \
        / np.prod(ricci_tensor.shape[1:])

    return {'einstein_norm': jnp.mean(einstein_tensor_norm * weights), 
            'F_functional': F_functional,
            'monge_ampere_loss': jnp.mean(ma_loss * weights),
            'ricci_tensor_norm': ricci_tensor_norm, 'chi_form': chi_form,
            'kahler_loss': k_loss.mean(), 'ricci_scalar': R.mean(), 'det_g': det_g_pred.mean(),
            'vol_loss': v_loss.mean(), 'vol_CY': vol_CY, 'vol_Omega': vol_Omega,
            'sigma_measure': _sigma_measure, 'ricci_measure': _ricci_measure}