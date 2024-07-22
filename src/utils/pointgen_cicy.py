import os, multiprocessing
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count())
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import jax

from jax import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

import numpy as np  # original CPU-backed NumPy
import jax.numpy as jnp

from jax import jit, jacfwd, vmap, random, pmap

from functools import partial

import math, time, argparse
import sympy as sp

import scipy.optimize as so
import joblib
from joblib import parallel_config, Parallel, delayed

# custom
from src import alg_geo
from src.utils import math_utils

class PointGenerator:
     
    def __init__(self, key: random.PRNGKey, cy_dim: int, monomials: np.array, 
                 coefficients: np.array, ambient: np.array):
        """
        Finds zero locus of a finite number of homogeneous polynomials in
        product of projective spaces. 
        
        Note: point generation significantly faster as standalone functions 
        but needs to be wrapped in a class for 'joblib' to execute correctly
        in a script ...
        """
        self.key = key
        self.cy_dim = cy_dim
        self.monomials, self.coefficients = monomials, coefficients
        self.ambient = ambient
        self.c_dim = cy_dim + 2  # homo. coords plus hypersurface constraint
        self.degrees = ambient + 1
        self.n_hyper = len(monomials)
        self.n_fold = np.sum(ambient) -self.n_hyper
        self.n_devs = len(jax.devices('cpu'))
        self.n_coords = monomials[0].shape[1]
        self.conf_mat, p_conf_mat = self._configuration_matrix(monomials, ambient) 
        self.t_degrees = self._find_degrees(ambient, self.n_hyper, self.conf_mat)
        self.kmoduli_ambient = math_utils._kahler_moduli_ambient_factors(self.cy_dim, self.ambient, self.t_degrees)
        self.METHOD = 'lm'
        self.all_monos, self.all_coeffs = jnp.concatenate(monomials), jnp.concatenate(coefficients)

        self.dQdz_info = [alg_geo.dQdz_poly(self.n_coords, m, c) for (m,c) in zip(monomials, coefficients)]
        self.dQdz_monomials, self.dQdz_coeffs = list(zip(*self.dQdz_info))
        self.all_dQdz_monos, self.all_dQdz_coeffs = jnp.concatenate(self.dQdz_monomials, axis=1), jnp.concatenate(self.dQdz_coeffs, axis=1)

        hyperplane_eqs, poly_gens = self._hyperplanes_symbolic(self.n_coords, self.n_hyper, self.ambient, self.t_degrees)

        self.t_monomials, self.t_coeffs = self.t_polynomial_data(hyperplane_eqs, poly_gens,
                                                monomials, coefficients, self.n_hyper)
        
        self.pdt_monomials, self.pdt_coeffs = self.t_polynomial_jacobian_data(self.t_monomials, self.t_coeffs,
                                                            self.n_coords, self.n_hyper)
        
    def parallel_sampling(self, key, n_p, PARAM_SCALE=0.5, LOCUS_TOL=1e-10):
        """
        Finds zero locus of a system of polynomial equations by solving for
        points of intersection with hyperplanes.
        """

        B = int(np.ceil(n_p / self.n_devs))
        key, key_, key__ = random.split(key, 3)
        pn_pts = self.sample_sphere_cicy(key_, n_p)
        params = jax.random.normal(key__, shape=(n_p, 2*self.n_hyper)) * PARAM_SCALE
        
        with Parallel(backend='multiprocessing', batch_size=B, verbose=5, n_jobs=-1) as parallel:
            z = parallel(delayed(self.poly_root)(par, p) for (par, p) in zip(params, pn_pts))  
       
        res = math_utils.to_complex(np.stack(z, axis=0))
        cicy_pts = vmap(self._point_from_sol)(pn_pts, res)
        abs_poly_val = jnp.abs(vmap(self.check_cicy_condition)(cicy_pts))
        cicy_pts = cicy_pts[abs_poly_val < LOCUS_TOL]

        return cicy_pts

    def sample_intersect_cicy(self, key: random.PRNGKey, n_p: int):

        LIMIT_RETRIES = 8
        FAILURE_RATE = 0.005
        print(f'Generating {n_p} points ...')
        output_pts = np.zeros((n_p, self.n_coords), dtype=np.complex128)
        key, key_ = random.split(key, 2)
        # sample a bit more to account for optimization failures
        n_excess = int(n_p * FAILURE_RATE + 10)
        cicy_pts = self.parallel_sampling(key_, n_p + n_excess)
        n_found = min(cicy_pts.shape[0], n_p)
        output_pts[:n_found] = cicy_pts[:n_found]
        retries = 0

        while n_found < n_p: # repeat until all points satisfy CICY condition
            retries += 1
            n_retry = int(np.ceil(n_p / n_found * (n_p - n_found)) + 42)
            print(f'Resampling {n_retry} points. Retries {retries}/(max.) {LIMIT_RETRIES}')
            key, key_ = random.split(key, 2)
            _cicy_pts = self.parallel_sampling(key_, n_p)
            n_good = _cicy_pts.shape[0]
            output_pts[n_found:n_found + n_good] = _cicy_pts[:(n_p - n_found)]
            n_found += n_good

            if retries >= LIMIT_RETRIES: break

        p = self._rescale_points(output_pts)
        abs_poly_val = jnp.abs(vmap(self.check_cicy_condition)(p)).max()
        print(f'Max locus violation: {abs_poly_val:.5e}')
        return p

    @partial(jit, static_argnums=(0,))
    def t_poly_optimize(self, params, p, root=True):
        """
        Numerically optimize for the roots of these polynomial(s) over the parameters
        of the hyperplane(s) to find the zero locus of the defining polynomial(s).
        Note this function is holomorphic.

        Args:
            p (ndarray[(ncoords, t-max-deg), np.complex128]): Values 
                for points on the spheres p, q, ...
            params (ndarray[2*n_hyper, np.float64]): t-values

        Returns:
            ndarray[..., 2*n_hyper, np.float64]: Difference from zero.
        """
        params = math_utils.to_complex(params)
        p_and_params = jnp.concatenate([p.reshape(-1), params], axis=-1)
        error = jnp.stack([alg_geo.evaluate_poly(p_and_params, t_mono, t_coeff) for (t_mono, t_coeff) in 
                    zip(self.t_monomials, self.t_coeffs)], axis=-1)
        
        if root is True: return math_utils.to_real(error)
        
        error = jnp.sum(jnp.abs(error))
        return error
    
    @partial(jit, static_argnums=(0,))
    def t_poly_jacobian(self, params, p, complex_jac=False):
        """
        Analytical Jacobian of objective function for root-finding routines. 
        Only use with 'vmap'
        
        Returns:
            ndarray[n_hyper, n_params], np.complex128]: Jacobian of each of the
            `n_hyper` polynomials w.r.t. parameters governing intersection.
        """
        dim = self.n_hyper * 2
        params = math_utils.to_complex(params)
        big_p = jnp.concatenate((p.reshape(-1), params))
        dT = jnp.stack([alg_geo.evaluate_poly(big_p, pdm, pdc) for
                pdm, pdc in zip(self.pdt_monomials, self.pdt_coeffs)], axis=0)
        
        if complex_jac is True: return jnp.squeeze(dT)

        # use the fact that polynomials are holomorphic and 
        # Cauchy-Riemann to calculate (deficient) Jacobian
        dudx, dudy = jnp.real(dT), -jnp.imag(dT)
        dT = jnp.zeros((dim,dim))
        c_dim = dim // 2
        dT = dT.at[:c_dim, :c_dim].set(dudx)
        dT = dT.at[:c_dim, c_dim:].set(dudy)
        dT = dT.at[c_dim:, :c_dim].set(-dudy)
        dT = dT.at[c_dim:, c_dim:].set(dudx)
        return dT
    

    @jit
    def t_poly_jacobian_autodiff(self, params, p):
        return jax.jacrev(self.t_poly_optimize)(params, p)

    def poly_root(self, params, p):
        return so.root(self.t_poly_optimize, 
                       params, p, 
                       jac=self.t_poly_jacobian, 
                       method=self.METHOD).x

    def _find_degrees(self, ambient, n_hyper, conf_mat):
        r"""Generates t-degrees in ambient space factors.
        Determines the shape for the expanded sphere points.
        """
        degrees = np.zeros(len(ambient), dtype=np.int32)
        for j in range(n_hyper):
            d = np.argmax(conf_mat[j])
            if degrees[d] == ambient[d]:
                # in case we already exhausted all degrees of freedom
                # shouldn't really be here other than for
                # some interesting p1 splits (redundant CICY description
                d = np.argmax(conf_mat[j, d + 1:])
            degrees[d] += 1
            
        return degrees

    def _configuration_matrix(self, monomials, ambient):
        conf_mat, n_monomials = [], []

        for m in monomials:
            n_monomials += [m.shape[0]]
            deg = []
            for i in range(len(ambient)):
                s = np.sum(ambient[:i]) + i
                e = np.sum(ambient[:i + 1]) + i + 1
                deg += [np.sum(m[0, s:e])]
            conf_mat += [deg]

        p_conf_mat = [[a] + c for a, c in zip(ambient, np.array(conf_mat).transpose().tolist())]

        return conf_mat, p_conf_mat

    def _hyperplanes_symbolic(self, n_coords, n_hyper, ambient, t_degrees):
        
        params = sp.symarray('t', n_hyper)
        ps = sp.symarray('p', (n_coords, max(t_degrees) + 1))
        poly_gens = list(ps.reshape(-1)) + list(params)
        
        for i in range(len(ambient)):
            for j in range(max(t_degrees) + 1):
                if j > t_degrees[i]:
                    s = np.sum(ambient[:i]) + i
                    e = np.sum(ambient[:i+1]) + i + 1
                    ps[s:e, j] = 0.
                    
        k = 0
        ts = sp.ones(n_coords, max(t_degrees) + 1)
        # define suitable free parameters for each hyperplane
        for i in range(len(ambient)):
            for j in range(t_degrees[i]):
                s = np.sum(ambient[:i]) + i
                e = np.sum(ambient[:i+1]) + i + 1
                ts[s:e, 1+j] = params[k] * sp.ones(*np.shape(ts[s:e, 1+j]))
                k += 1

        hyperplane_eqs = np.sum(np.array(ts) * ps, axis=-1)
        return hyperplane_eqs, sorted(poly_gens, key=str)

    def t_polynomial_data(self, hyperplane_eqs, poly_gens, monomials, coefficients, n_hyper):
        t_poly = [alg_geo.evaluate_poly_onp(hyperplane_eqs, monos, coeffs) for 
                    (monos, coeffs) in zip(monomials, coefficients)]
        
        # express as polynomial in sampled points and params `t`
        poly_dict = [sp.Poly(tp, poly_gens).as_dict() for tp in t_poly]
        t_exponents = [np.zeros((len(terms), len(poly_gens)), dtype=np.int32) for terms in poly_dict] 
        t_coeffs = [np.zeros(len(terms), dtype=np.complex128) for terms in poly_dict]
        for i in range(n_hyper):    
            for j, (exponents, coeffs) in enumerate(poly_dict[i].items()):
                t_exponents[i][j] = exponents
                t_coeffs[i][j] = coeffs

        return t_exponents, t_coeffs

    def t_polynomial_jacobian_data(self, t_monomials, t_coeffs, n_coords, n_hyper):
        # Get partial derivative of t-polynomial w.r.t. parameters for use
        # in Jacobian during optimization.
        n_all_coords = n_coords * (max(self.t_degrees) + 1) + n_hyper
        dTdz_polys = [alg_geo.dQdz_poly(n_all_coords, m, c) for m,c in zip(t_monomials, t_coeffs)]
        pdt_monomials = [dT[0][-n_hyper:] for dT in dTdz_polys]
        pdt_coeffs = [dT[1][-n_hyper:] for dT in dTdz_polys]
        return pdt_monomials, pdt_coeffs

    def sample_sphere_cicy(self, key, n_p):
        max_deg = max(self.t_degrees)
        keys = random.split(key, len(self.ambient)*(max_deg+1))
        pn_pts = jnp.zeros((n_p, self.n_coords, max_deg+1), dtype=np.complex128)
        for i in range(len(self.ambient)):
            for k in range(self.t_degrees[i] + 1):
                key_idx = i*(max_deg+1)+k
                s = jnp.sum(self.ambient[:i]) + i
                e = jnp.sum(self.ambient[:i+1]) + i + 1
                pn_pts = pn_pts.at[:, s:e, k].set(
                    jnp.squeeze(math_utils.S2np1_uniform(keys[key_idx], n_p, self.ambient[i])))
        return pn_pts
    
    def _point_from_sol(self, p, sol):
        r"""Generates a point on the CICY. Use with `vmap`.

        Args:
            p (ndarray[(ncoords, t-max-deg), np.complex128]): Values 
                for points on the spheres p, q, ...
            sol (ndarray[(nHyper), np.complex]): Complex t-values.

        Returns:
            ndarray[(ncoords), np.complex128]: Point on the (CI-)CY.
        """
        # use this over point from sol sympy >100 factor improvement
        t = jnp.ones_like(p)
        k = 0
        for i in range(len(self.ambient)):
            for j in range(1, self.t_degrees[i] + 1):
                s = jnp.sum(self.ambient[:i]) + i
                e = jnp.sum(self.ambient[:i + 1]) + i + 1
                t = t.at[s:e, j].set(sol[k] * jnp.ones_like(t[s:e, j]))
                k += 1
        point = jnp.sum(p * t, axis=-1)
        return point

    @partial(jit, static_argnums=(0,))
    def check_cicy_condition(self, p):
        r"""Computes the CICY condition at each point.

        Args:
            p:  (ndarray[(n_p, n_coords), np.complex128]): Points nominally on CICY.
            monomials:  (ndarray[(n_hyper * (n_ambient + 1), n_coords),
                                    np.int32]): Exponents of all defining monomials.
            coefficients:  (ndarray[(n_hyper * (n_ambient + 1)), np.complex128]):
                            Coefficients of all defining polynomials.
        Returns:
            ndarray([n_p, n_hyper], np.complex128): CICY condition
        """
        return alg_geo.evaluate_poly(p, self.all_monos, self.all_coeffs)

    def _argmax_dQdz(self, p, dQdz=None, aux=False):
        """
        Finds suitable coordinates to eliminate for a submanifold embedded in a product of
        projective spaces by iteratively searching for \argmax_i \vert z_i \vert and eliminating.
        Also removes coords rescaled to unity. 
        """
        inv_ones_mask = jnp.isclose(p, jax.lax.complex(1.,0.))
        total_mask = jnp.logical_not(inv_ones_mask)

        if dQdz is None:
            dQdz = alg_geo.evaluate_dQdz(p, self.dQdz_monomials, self.dQdz_coeffs)

        elim_idx = jnp.zeros(self.n_hyper, dtype=np.int32)
        for i in range(self.n_hyper):
            elim_idx_i = jnp.argmax(jnp.abs(dQdz[i] * total_mask), axis=-1).astype(np.int32)
            elim_idx = elim_idx.at[i].set(elim_idx_i)
            elim_mask_i = (jnp.arange(self.n_coords) == elim_idx_i[np.newaxis])
            total_mask *= ~elim_mask_i

        if aux is True:
            return elim_idx, total_mask

        return elim_idx


    def holomorphic_volume_form(self, p):

        dQdz = alg_geo.evaluate_dQdz(p, self.dQdz_monomials, self.dQdz_coeffs)

        elim_idx = jnp.expand_dims(self._argmax_dQdz(p, dQdz), axis=-1)
        residues = jnp.squeeze(jnp.take_along_axis(dQdz, elim_idx, axis=1))
        residues = jnp.prod(residues, axis=-1)

        return 1./residues
    
    def _rescale_points(self, p):
        """
        Rescales points 'p' s.t. (\max_i \abs{p} == 1.0) in each projective
        space factor.
        """
        for i in range(len(self.ambient)):
            s = np.sum(self.degrees[0:i])
            e = np.sum(self.degrees[0:i+1])
            p[:, s:e] = math_utils.rescale(p[:, s:e])[0]
        
        return p

def t_poly_optimize(params, p, t_monomials, t_coeffs, root=True):
    """
    Numerically optimize for the roots of these polynomial(s) over the parameters
    of the hyperplane(s) to find the zero locus of the defining polynomial(s).
    Note this function is holomorphic.

    Args:
        p (ndarray[(ncoords, t-max-deg), np.complex128]): Values 
            for points on the spheres p, q, ...
        params (ndarray[2*n_hyper, np.float64]): t-values

    Returns:
        ndarray[..., 2*n_hyper, np.float64]: Difference from zero.
    """
    params = math_utils.to_complex(params)
    p_and_params = jnp.concatenate([p.reshape(-1), params], axis=-1)
    error = jnp.stack([alg_geo.evaluate_poly(p_and_params, t_mono, t_coeff) for (t_mono, t_coeff) in 
                 zip(t_monomials, t_coeffs)], axis=-1)
    
    if root is True: return math_utils.to_real(error)
    
    error = jnp.sum(jnp.abs(error))
    return error

def _t_poly_jacobian(params, p, t_monomials, t_coeffs):
    """
    Internal use only. Autodiff Jacobian of `t_poly_optimize`. 
    Use with `vmap`.
    Note: inefficient.
    Returns:
        ndarray[n_hyper, n_params], np.complex128]: Jacobian of each of the
        `n_hyper` polynomials w.r.t. parameters governing intersection.
    """
    tpo = partial(t_poly_optimize, t_monomials=t_monomials, t_coeffs=t_coeffs)
    jac = jax.jacrev(tpo, 0)(params, p)  # params should be real
    param_dim = jac.shape[-1]
    c_param_dim = param_dim // 2
    dTdx, dTdy = jac[..., :c_param_dim], jac[..., c_param_dim:]
    dTdz = 0.5 * (dTdx - 1.j * dTdy)  # del_z w.r.t. params

    # combine Re, Im parts of output of `t_poly_optimize`
    out_dim = dTdz.shape[0]
    c_out_dim = out_dim // 2
    dTdz = dTdz[..., :c_out_dim, :] + 1.j * dTdz[..., c_out_dim:, :]
    return dTdz

def t_poly_jacobian(params, p, pd_monomials, pd_coeffs, dim, 
                    complex_jac=False):
    """
    Analytical Jacobian of objective function for root-finding routines. 
    Only use with 'vmap'
    
    Returns:
        ndarray[n_hyper, n_params], np.complex128]: Jacobian of each of the
        `n_hyper` polynomials w.r.t. parameters governing intersection.
    """
    params = math_utils.to_complex(params)
    big_p = jnp.concatenate((p.reshape(-1), params))
    dT = jnp.stack([alg_geo.evaluate_poly(big_p, pdm, pdc) for
            pdm, pdc in zip(pd_monomials, pd_coeffs)], axis=0)
    
    if complex_jac is True: return jnp.squeeze(dT)

    # use the fact that polynomials are holomorphic and 
    # Cauchy-Riemann to calculate (deficient) Jacobian
    dudx, dudy = jnp.real(dT), -jnp.imag(dT)
    dT = jnp.zeros((dim,dim))
    c_dim = dim // 2
    dT = dT.at[:c_dim, :c_dim].set(dudx)
    dT = dT.at[:c_dim, c_dim:].set(dudy)
    dT = dT.at[c_dim:, :c_dim].set(-dudy)
    dT = dT.at[c_dim:, c_dim:].set(dudx)
    return dT
    

def t_poly_jacobian_onp_single(params, p, pd_monomials, pd_coeffs, complex_jac=False):
    """
    Numpy version for lower per-dispatch overhead?
    """
    dim = params.shape[-1]
    c_dim = dim // 2
    params = params[:c_dim] + 1.j * params[c_dim:]
    big_p = np.concatenate((p.reshape(-1), params), axis=-1)
    dT = np.stack([alg_geo.evaluate_poly_batch(np.expand_dims(big_p,0),
                                               pdm, pdc) for
            pdm, pdc in zip(pd_monomials, pd_coeffs)], axis=0)
    if complex_jac is True: return np.squeeze(dT)

    # use the fact that polynomials are holomorphic and 
    # Cauchy-Riemann to calculate (deficient) Jacobian
    dudx, dudy = np.real(dT), -np.imag(dT)
    dT = np.zeros((dim,dim))
    dT[:c_dim, :c_dim] = dudx
    dT[:c_dim, c_dim:] = dudy
    dT[c_dim:, :c_dim] = -dudy
    dT[c_dim:, c_dim:] = dudx
    return dT

def t_poly_jacobian_onp_batch(params, p, pd_monomials, pd_coeffs, complex_jac=True):
    """
    Numpy version for lower per-dispatch overhead.
    """
    B, dim = params.shape
    c_dim = dim // 2
    params = params[...,:c_dim] + 1.j * params[...,c_dim:]
    big_p = np.concatenate((p.reshape(B,-1), params), axis=-1)
    dT = np.stack([alg_geo.evaluate_poly_batch(np.expand_dims(big_p,1),
                                               pdm, pdc) for
            pdm, pdc in zip(pd_monomials, pd_coeffs)], axis=1)
    
    if complex_jac is True: return np.squeeze(dT)
    
    # use the fact that polynomials are holomorphic and 
    # Cauchy-Riemann to calculate (redundant) Jacobian
    dudx, dudy = np.real(dT), -np.imag(dT)
    dT = np.zeros((B,dim,dim))
    dT[:, :c_dim, :c_dim] = dudx
    dT[:, :c_dim, c_dim:] = dudy
    dT[:, c_dim:, :c_dim] = -dudy
    dT[:, c_dim:, c_dim:] = dudx
    return dT

def t_poly_optimize_onp(params, p, t_monomials, t_coeffs, root=True):
    """
    Numerically optimize for the roots of these polynomial(s) over the parameters
    of the hyperplane(s) to find the zero locus of the defining polynomial(s).
    Note this function is holomorphic.

    Args:
        p (ndarray[(ncoords, t-max-deg), np.complex128]): Values 
            for points on the spheres p, q, ...
        params (ndarray[2*n_hyper, np.float64]): t-values

    Returns:
        ndarray[..., 2*n_hyper, np.float64]: Difference from zero.
    """
    params = math_utils.to_complex(params)
    p_and_params = np.concatenate([p.reshape(-1), params], axis=0)
    error = np.stack([alg_geo.evaluate_poly(p_and_params, t_mono, t_coeff) for (t_mono, t_coeff) in 
                 zip(t_monomials, t_coeffs)], axis=-1)
    
    if root is True: return math_utils.to_real(error)
    
    error = np.sum(np.abs(error))
    return error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='CICY point generation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-o', '--output_path', type=str, help="Path to the output directory for points.", required=True)
    parser.add_argument('-n_p', '--num_pts', type=int, help="Number of points to generate.", default=100000)
    parser.add_argument('-val', '--val_frac', type=float, help="Percentage of points to use for validation.", default=0.2)
    args = parser.parse_args()

    from tqdm import tqdm

    from src import fubini_study, dataloading
    from src.utils import gen_utils as utils
    from examples import poly_spec

    start = time.time()
    key = random.PRNGKey(int(start))
    n_devs, n_p = len(jax.devices('cpu')), args.num_pts
    v_p = int(args.val_frac * n_p)

    # Specify polynomial here
    # ========================
    poly_specification = poly_spec.tian_yau_KM_spec
    coeff_fn = poly_spec.tian_yau_KM_coefficients
    psi = -0.75
    coefficients = coeff_fn(psi)
    # ========================

    monomials, cy_dim, kmoduli, ambient = poly_specification()
    pg_cicy = PointGenerator(key, cy_dim, monomials, coefficients, ambient)
    dQdz_monomials, dQdz_coeffs = pg_cicy.dQdz_monomials, pg_cicy.dQdz_coeffs
    
    # generate points
    cicy_pts = pg_cicy.sample_intersect_cicy(key, n_p + v_p)
    
    if pg_cicy.n_hyper == 1:
        get_metadata = partial(alg_geo.compute_integration_weights, cy_dim=cy_dim)
        det_g_FS_fn = fubini_study.det_fubini_study_pb
    else:
        get_metadata = partial(alg_geo._integration_weights_cicy, n_hyper=pg_cicy.n_hyper, cy_dim=cy_dim, 
                               n_coords=pg_cicy.n_coords, ambient=pg_cicy.ambient, kmoduli_ambient=pg_cicy.kmoduli_ambient)
        det_g_FS_fn = partial(fubini_study.det_fubini_study_pb_cicy, n_coords=pg_cicy.n_coords,
                ambient=tuple(pg_cicy.ambient), cdtype=np.complex128)
    get_metadata = jit(get_metadata)

    B = (n_p + v_p) // 12
    data_batched = dataloading._online_batch(cicy_pts, n_p, B)
    weights, pullbacks, dVol_Omegas = [], [], []
    vol_Omega, vol_g = 0., 0.
    
    # TODO: Add hypersurface support to 'pointgen'
    for data in tqdm(data_batched):
        _p = data
        w, pb, _dVol_Omega, *_ = vmap(get_metadata, in_axes=(0,None,None))(_p, dQdz_monomials, dQdz_coeffs)
        weights.append(w)
        pullbacks.append(pb)
        dVol_Omegas.append(_dVol_Omega)
    
        # compute Monge-Ampere proportionality constant
        _det_g_FS_pb = vmap(det_g_FS_fn)(math_utils.to_real(_p), pb)
        _vol_g = jnp.mean(w * _det_g_FS_pb / _dVol_Omega).item()
        vol_g = math_utils.online_update(vol_g, _vol_g, n_p, B)

        _vol_Omega = jnp.mean(w).item()
        vol_Omega = math_utils.online_update(vol_Omega, _vol_Omega, n_p, B)

    weights, pullbacks = np.squeeze(np.concatenate(weights, axis=0)), np.squeeze(np.concatenate(pullbacks, axis=0))
    dVol_Omegas = np.squeeze(np.concatenate(dVol_Omegas, axis=0))

    kappa = vol_g / vol_Omega
    print(f'kappa: {kappa:.7f}')

    print(f'Saving under {args.output_path}/ ...')
    os.makedirs(args.output_path, exist_ok=True)
    f = os.path.join(args.output_path, 'dataset.npz')

    p = math_utils.to_real(cicy_pts)

    if args.val_frac == 0.:
        np.savez_compressed(f, x=p, w=weights, pb=pullbacks, dVol_Omega=dVol_Omegas, 
                            kappa=kappa, vol_g=vol_g, vol_Omega=vol_Omega, psi=psi)
    else:
        p_train, p_val = np.array_split(p, (n_p,))
        w_train, w_val = np.array_split(weights, (n_p,))
        dVol_Omega_train, dVol_Omega_val = np.array_split(dVol_Omegas, (n_p,))
        pb_train, pb_val = np.array_split(pullbacks, (n_p,))

        y_train = np.stack((w_train, dVol_Omega_train), axis=-1)
        y_val = np.stack((w_val, dVol_Omega_val), axis=-1)

        np.savez_compressed(f, x_train=p_train, y_train=y_train, x_val=p_val, 
                            y_val=y_val, pb_train=pb_train, pb_val=pb_val, 
                            kappa=kappa, vol_g=vol_g, vol_Omega=vol_Omega, psi=psi)
        
    metadata = utils.save_metadata(poly_specification(), coefficients, kappa, args.output_path)
    delta_t = time.time() - start
    print(f'Time elapsed: {delta_t:.3f} s')
