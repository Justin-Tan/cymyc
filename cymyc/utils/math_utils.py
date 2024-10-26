import jax
import math
import itertools
import numpy as np
import jax.numpy as jnp

from jax import jit, random, vmap
from functools import partial


@jit
def to_complex(x):
    """
    Reshapes 2m-dim real vector `x` to m-dim complex vector, 
    where `x` = [Re(z) | Im(z)] <- divided into halves 
    """
    # assert x.shape[-1] % 2 == 0
    c_dim = x.shape[-1] // 2
    return jax.lax.complex(x[...,0:c_dim], x[...,c_dim:])

@jit
def to_real(z):
    """
    Reshapes m-dim complex vector to 2m-dim real vector
    z -> [x; y]= [Re(z); Im(z)] 
    """
    complex_dim = z.shape[-1]
    if not jnp.issubdtype(z.dtype, jnp.complexfloating):
        return z
    
    re_z = z.real
    xy = jnp.zeros(z.shape+(2,), dtype=re_z.dtype)
    xy = xy.at[...,0].set(re_z)
    xy = xy.at[...,1].set(z.imag)
    
    xy = xy.reshape(-1, complex_dim, 2)
    xy = jnp.concatenate(jnp.split(xy, 2, axis=-1), axis=1)
    return jnp.squeeze(xy)

def to_real_tensor(z):
    """
    Don't `vmap`, expects batch dimension [B,n1,...,nk]
    """

    complex_dim = np.prod(z.shape[1:])
    if not jnp.issubdtype(z.dtype, jnp.complexfloating):
        return z
    
    re_z = z.real
    xy = jnp.zeros(z.shape+(2,), dtype=re_z.dtype)
    xy = xy.at[...,0].set(re_z)
    xy = xy.at[...,1].set(z.imag)
    return jnp.squeeze(xy)

def to_complex_tensor(x):
    return jax.lax.complex(x[...,0], x[...,1])


def to_real_onp(z):
    B, complex_dim = z.shape[0], z.shape[-1]
    xy = np.zeros(z.shape + (2,), dtype=np.float64)
    xy[...,0] = np.real(z)
    xy[...,1] = np.imag(z)
    xy = xy.reshape(B, -1, complex_dim*2)
    return xy

def max_n_derivs(monomials):
    return max([len(np.nonzero(monomials[:,i])[0]) for i in range(monomials.shape[-1])])

def get_valid_dQ_idx(monomials, n_coords):
    return np.nonzero([len(np.nonzero(monomials[:,i])[0]) for i in range(n_coords)])[0]

def get_valid_lims(monomials):
    return np.unique(np.nonzero(monomials)[-1], return_counts=True)[-1]

def _inhomogenize(p):
    """
    Converts 'p' (n+1, np.complex128) homogeneous coords in P^n to n inhomogeneous coords.
    Assumes `p` has been rescaled s.t. \argmax_i \abs{p_i} = 1.0.
    """
    mask = jnp.logical_not(jnp.isclose(p, jax.lax.complex(1.,0.)))
    p_inhomo = p[jnp.nonzero(mask, size=p.shape[-1]-1)]
    return p_inhomo

def _find_degrees(ambient, n_hyper, conf_mat):
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

def _kahler_moduli_ambient_factors(cy_dim, ambient, t_degrees):
    all_omegas = jnp.array(ambient - t_degrees)
    ts = jnp.zeros((cy_dim, len(all_omegas)), dtype=np.int32)
    j = 0
    for i in range(len(ambient)):
        for _ in range(all_omegas[i]):
            ts = ts.at[j,i].set(ts[j,i]+1)
            j += 1
    return ts

def _configuration_matrix(monomials, ambient):
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

def online_update(mu, x, n, B=1., S=None, _S=None):
    """
    Use's Welford's method to compute the running variance, if 
    `S` provided.
    """
    Z_SCORE_THRESHOLD = 3
    FREE_START = 3
    if n == 0: mu = x

    running_mean = mu + (x - mu) * B/(n+B)

    if S is not None:
        if jnp.isnan(x): return mu, S
        delta = mu - x
        running_S = S + _S + delta**2 * n * B / (n+B)
        var = S / (n-1)
        if n <= FREE_START * B:
            return running_mean, running_S
        if ((jnp.abs(mu - x) / jnp.sqrt(var)) > Z_SCORE_THRESHOLD):
            return mu, S
        return running_mean, running_S

    return running_mean

@jit
def online_update_array(mu, x, n, B=1., S=None, _S=None):
    """
    Use's Welford's method to compute the running variance, if 
    `S` provided.
    """
    Z_SCORE_THRESHOLD = 3
    FREE_START = 5
    mu = jnp.where(n==0, x, mu)
    mu_update = mu + (x - mu) * B / (n+B)
    # x = jnp.where(jnp.isnan, mu, x)

    if S is not None:
        delta = mu - x
        S_update = S + _S + delta**2 * n * B / (n+B)

        # reject large deviations
        var = S / (n-1)
        mask = (jnp.abs(mu - x) / jnp.sqrt(var)) < Z_SCORE_THRESHOLD

        running_mean = jnp.where(mask, mu_update, mu)
        running_S = jnp.where(mask, S_update, S)

        running_mean = jnp.where(n <= FREE_START * B, mu_update, running_mean)
        running_S = jnp.where(n <= FREE_START * B, S_update, running_S)
        
        return running_mean, running_S

    return mu_update

def shifted_variance(x, shift):
    n = x.shape[0]
    Ex = jnp.sum(x-shift, axis=0)
    Ex2 = jnp.sum(jnp.square(x-shift), axis=0)
    S = (Ex2 - jnp.square(Ex)/n)
    return S

def unsqueeze(x):
    # Use to add batch dimension to single examples
    return np.expand_dims(x, 0)

def complex_mult(u,v,x,y):
    return u*x + v*y, v*x - u*y

def rescale(x):
    """
    Convert (n+1)-dim homogeneous coords to n-dim inhomogeneous coords by dividing by
    complex coordinate with maximum modulus.
    """
    m = jnp.argmax(jnp.abs(x), axis=-1)
    x = x / jnp.take_along_axis(x, jnp.expand_dims(m,-1), axis=-1)    
    return x, m

def S2np1_uniform(key, n_p, n):
    """
    Sample `n_p` points uniformly on $S^{2n+1}$, treated as CP^n
    """
    # return random.uniform(key, (n,))*jnp.pi, random.uniform(key, (n,))*2*jnp.pi
    x = random.normal(key, shape=(n_p, 2*(n+1)))
    x_norm = x / jnp.linalg.norm(x, axis=1, keepdims=True)
    sample = to_complex(x_norm.reshape(-1, n+1, 2))
    
    return sample

# @partial(jit, static_argnums=(1,2))
def inhomogenize_batch(rng, ambient_dim, n_samples=10000):
    """
    Returns sample in inhomogeneous coordinates from CP^n by
    converting (n+1)-dim homogeneous coords to n-dim inhomogeneous coords
    """
    Pz = jnp.squeeze(S2np1_uniform(rng, n_samples, ambient_dim))
    x, m = vmap(rescale)(Pz)
    mask = jnp.isclose(jnp.real(x), 1.)
    x = x[~mask].reshape(-1, ambient_dim) # bit dodgy...
    return x

def epsilon_symbol(*args):
    """
    Returns value of n-dim epsilon symbol at indices defined by
    args, an n-dim iterable of indices
    """
    n = len(args)
    return np.prod(
        [np.prod([args[j] - args[i] for j in range(i + 1, n)])
        / math.factorial(i) for i in range(n)])

def n_dim_eps_symbol(n):
    """
    Constructs n^n size array corresponding to epsilon symbol -
    very inefficient! Probably a better way using bitmasking
    for tensor contractions...
    """
    eps_sym_nd = np.zeros([n]*n)
    for idx in itertools.permutations([i for i in range(n)]):
        eps_sym_nd[idx] = epsilon_symbol(*idx)
    return eps_sym_nd

@jit
def eps_2D_contract(eps_2d,x,y):
    contraction = jnp.einsum('...ij, ...ia, ...jb, ...ab -> ...', eps_2d, x, y, eps_2d)
    return contraction

@jit
def eps_3D_contract(eps_3d,x,y,z):
    contraction = jnp.einsum('...ijk, ...ia, ...jb, ...kc, ...abc -> ...', eps_3d, x, y, z, eps_3d)
    return contraction

@jit
def eps_4D_contract(eps_4d,w,x,y,z):
    contraction = jnp.einsum('...ijkl, ...ia, ...jb, ...kc, ...ld, ...abcd -> ...', eps_4d, w, x, y, z, eps_4d)
    return contraction

# @partial(jit, static_argnums=(1,))
def log_det_fn(p, g, *args):
   
    det_g = jnp.real(jnp.linalg.det(g(p, *args)))
    return jnp.log(det_g)

def Pi(conf, kmoduli):
    import sympy, functools
    k, l = conf.shape[0], conf.shape[1]-1
    Pns, degrees = conf[:,0], conf[:,1:]
    prefactor = np.prod([math.factorial(n_i) for n_i in Pns])

    Js = " ".join([f"J_{i}" for i in range(k)])
    Js = np.array(sympy.symbols(Js))
    _Js = np.expand_dims(Js, len(np.array(Js).shape))

    ts = " ".join([f"t_{i}" for i in range(k)])
    ts = np.array(sympy.symbols(ts))
    
    c_top_NX = np.prod(np.sum(degrees * _Js, axis=0))
    c_X = (np.prod(np.power(np.array(1) + Js, Pns+1))) / np.prod(1 + np.sum(degrees * _Js, axis=0))

    chi_terms = c_top_NX * c_X
    c2_w_J_terms = chi_terms * np.sum(Js * ts)
    vol_terms = chi_terms * np.power(np.sum(Js * ts), 3)
    
    if len(np.array(Js).shape) == 0:
        Js = np.expand_dims(Js, 0)

    _t = functools.reduce(sympy.diff, zip(Js, Pns), chi_terms)
    chi = _t.subs(list(zip(Js, np.zeros_like(Js)))) / prefactor
    assert type(chi) is sympy.core.numbers.Integer, 'Euler characteristic is not an integer!'
    
    _t = functools.reduce(sympy.diff, zip(Js, Pns), c2_w_J_terms)
    c2_w_J = _t.subs(list(zip(Js, np.zeros_like(Js)))) / prefactor

    _t = functools.reduce(sympy.diff, zip(Js, Pns), vol_terms)
    vol = _t.subs(list(zip(Js, np.zeros_like(Js)))) / prefactor

    if len(np.array(ts).shape) == 0:
        ts = np.expand_dims(ts, 0)
    if kmoduli is None: kmoduli = np.ones_like(ts)
    canonical_vol = vol.subs(list(zip(ts, kmoduli)))
    
    # return chi, sympy.simplify(c2_w_J), sympy.simplify(vol), canonical_vol
    return int(chi), str(sympy.simplify(c2_w_J)), str(sympy.simplify(vol)), float(canonical_vol)
