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
	#try: 
	#	 if mu == 0.: mu = x
	#except ValueError:
	if n == 0:	# this shouldn't happen except w/ legacy code
		mu = x
		n += B

	running_mean = mu + (x - mu) * B/(n+B)

	if S is not None:
		delta = mu - x
		running_S = S + _S + delta**2 * n * B / (n+B)
		var = running_S / (n-1)
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
	FREE_START = 2
	mu = jnp.where(n==0, x, mu)
	mu_update = mu + (x - mu) * B / (n+B)

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
	very inefficient! There is probably a better way using bitmasking
	for tensor contractions?
	"""
	eps_sym_nd = np.zeros([n]*n)
	for idx in itertools.permutations([i for i in range(n)]):
		eps_sym_nd[idx] = epsilon_symbol(*idx)
	return eps_sym_nd

eps_2d = jnp.array(n_dim_eps_symbol(2))
eps_3d = jnp.array(n_dim_eps_symbol(3))
eps_4d = jnp.array(n_dim_eps_symbol(4))

@jit
def eps_2D_contract(x,y):
	contraction = jnp.einsum('...ij, ...ia, ...jb, ...ab -> ...', eps_2d, x, y, eps_2d)
	return contraction

@jit
def eps_3D_contract(x,y,z):
	contraction = jnp.einsum('...ijk, ...ia, ...jb, ...kc, ...abc -> ...', eps_3d, x, y, z, eps_3d)
	return contraction

@jit
def eps_4D_contract(w,x,y,z):
	contraction = jnp.einsum('...ijkl, ...ia, ...jb, ...kc, ...ld, ...abcd -> ...', eps_4d, w, x, y, z, eps_4d)
	return contraction

# @partial(jit, static_argnums=(1,))
def log_det_fn(p, g, *args):
   
	det_g = jnp.real(jnp.linalg.det(g(p, *args)))
	return jnp.log(det_g)

def log_tan(x): return jnp.log(jnp.tan(x))
def constant(x): return 1.

class monte_carlo(object):

	@staticmethod
	@partial(jit, static_argnums=(1,2,3,4))
	def generateDomain(rng, a, b, dim, num_pts):
		x_unif = random.uniform(rng, shape=(num_pts, dim), 
								minval=a, maxval=b)
		vol = jnp.power(b - a, dim)
		return x_unif, vol

	@staticmethod
	@partial(jit, static_argnums=(1,2,3,4,5))
	def integrate(rng, func, a, b, dim, num_pts):
		"""
		integrate over the domain [a,b]^n
		TODO: Rejection sampling
		:param func: real/complex valued function.
		:param a: domain [a,b]^n specification variable.
		:param b: domain [a,b]^n specification variable.
		:param n: domain [a,b]^n specification variable.
		:param num_pts: number of points to sample.
		"""
		x, vol = monte_carlo.generateDomain(rng, a, b, dim, num_pts)
		y = vmap(func)(x)
		y_mean = jnp.mean(y)
		return vol * y_mean

	@staticmethod
	@partial(jit, static_argnums=(1,2,3,4,5,6,7))
	def integrate_Rn(rng, func, volume_form, dim, num_pts, 
					 a=None, b=None, transform=None):
		"""
		Convert unbounded integration domain over R^n to bounded domain D 
		by pullback via invertible, unbounded function f: D -> R
		"""
		def volume_form_times_func(p):
			return volume_form(p) * func(p)

		if transform is None: 
			a, b, transform = -np.pi/2, np.pi/2, jnp.tan
		elif transform is log_tan:
			a, b = 0, np.pi/2
		else:
			assert ((a is not None) and (b is not None)), 'Must provide limits!'
		
		x, vol = monte_carlo.generateDomain(rng, a, b, dim, num_pts)
		
		if transform is jnp.tan:
			jacobian_det = jnp.real(jnp.prod(1./(jnp.square(jnp.cos(x))), axis=-1))
		elif transform is log_tan:
			jacobian_det = jnp.real(jnp.prod(1./(jnp.cos(x)*jnp.sin(x)), axis=-1))
		else:
			jacobian_det = jnp.real(jnp.linalg.det(vmap(jax.jacfwd(transform))(x)))
		
		y = vmap(volume_form_times_func)(transform(x))

		return jnp.mean(y * jacobian_det * vol)
