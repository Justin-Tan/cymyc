"""
Specification of basis of polynomial deformations for c.s. deformation space 
specified in Kalara/Mohapatra (1987) 
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import jax
from jax import config
config.update("jax_enable_x64", False)

import numpy as np
import jax.numpy as jnp
from jax import jit

from functools import partial

# custom
from cymyc import alg_geo

# @jit
def poly_deformation(p: jax.typing.ArrayLike, monomials: jax.typing.ArrayLike, 
					 coefficients: jax.typing.ArrayLike, eq_idx: jax.typing.ArrayLike,
					 precision: jax.typing.DTypeLike = None):#jax.lax.Precision('default')):
	"""
	Deforms homogeneous polynomials in product of projective spaces
	whose common zero loci define the CY. 
	Parameters:
	`p`:			 Points on CY, shape [...,cy_dim]
	`monomials`:	 Monomial corresponding to CS deformation, shape [n_monomials, n_inhomo_coords]
	`coefficients`:  Multiplicative factor for each monomial, shape [n_monomials]
	`eq_idx`:		 Index of defining equation each deformation is applied to, shape [1, n_hyper]
	`precision`:	 Controls tradeoff between speed/accuracy. Fastest: jax.lax.Precision('default'),
					 most accurate: jax.lax.Precision('highest')
	
	Returns:
	`p_deform`:		 Polynomial deformation, shape [..., n_hyper].
	"""

	p_deform = alg_geo.evaluate_poly(p, monomials, coefficients)
	# choose which defining eq. to deform
	p_deform = jnp.einsum('...i,...ij->...j', jnp.expand_dims(p_deform, axis=-1), eq_idx,
						  precision=precision)
	return p_deform


def TY_KM_deformations(cdtype=np.complex64):
	"""
	Diagonal basis for $H^{(0,1)}(X,TX)$ after Gram-Schmidt using WP
	See Candelas, Kalara (1988), Nuclear Physics B298, Table 1.
	Assumes Z3 quotient.
	"""
	deformation_monomials = [
		np.asarray([[1, 1, 1, 0, 0, 0, 0, 0]]),  # λ1 - e1
		np.asarray([[1, 1, 0, 1, 0, 0, 0, 0]]),  # λ2

		np.asarray([[0, 0, 0, 0, 1, 1, 1, 0]]),  # λ3 - e2
		np.asarray([[0, 0, 0, 0, 1, 1, 0, 1]]),  # λ4

		np.asarray([[0, 1, 0, 0, 0, 1, 0, 0]]),  # λ5 - e3
		np.asarray([[0, 0, 1, 0, 0, 0, 1, 0]]),  # λ6
		np.asarray([[0, 0, 0, 1, 0, 0, 0, 1]]),  # λ7
		np.asarray([[0, 0, 1, 0, 0, 0, 0, 1]]),  # λ8
		np.asarray([[0, 0, 0, 1, 0, 0, 1, 0]]),  # λ9	
	]

	n_hyper = 3  # number of defining equations
	deformation_coeffs = 3 * np.eye(9, dtype=cdtype)

	basis = jnp.split(jnp.eye(n_hyper), n_hyper)
	deformation_idx = [basis[0]]*2 + [basis[1]]*2 + [basis[2]]*5  # TODO: choose to make singularity occur at \psi=1
	deformation_idx = jnp.concatenate(deformation_idx)

	dm = jnp.concatenate(deformation_monomials)
	dc = jnp.einsum('...i,ij->...ij', deformation_coeffs, deformation_idx)
	deformations = [partial(alg_geo.evaluate_poly, monomials=dm, coefficients=c.T) for c in dc]

	# Kalara-Mohapatra diagonal basis
	dm4 = [np.asarray([[0, 0, 3, 0, 0, 0, 0, 0]]),
		   np.asarray([[0, 0, 0, 3, 0, 0, 0, 0]])]
	dm4 = jnp.concatenate(dm4)
	dc4 = 3 * np.asarray([1.,1.], dtype=cdtype)  # corresponds to KM deformation direction

	dm5 = dm4
	dc5 = 3 * np.asarray([1.,-1.], dtype=cdtype)

	dm6 = [np.asarray([[0, 3, 0, 0, 0, 0, 0, 0]]),
		   np.asarray([[3, 0, 0, 0, 0, 0, 0, 0]])]
	dm6 = jnp.concatenate(dm6)
	dc6 = 3 * np.asarray([1.,-1.], dtype=cdtype)

	dm7 = np.asarray([[0, 0, 1, 2, 0, 0, 0, 0]])
	dc7 = 3 * np.asarray([1.], dtype=cdtype)

	dm8 = np.asarray([[0, 0, 2, 1, 0, 0, 0, 0]])
	dc8 = 3 * np.asarray([1.], dtype=cdtype)

	deformations[4] = partial(poly_deformation, monomials=dm4, coefficients=dc4, eq_idx=basis[0])
	deformations[5] = partial(poly_deformation, monomials=dm5, coefficients=dc5, eq_idx=basis[0])
	deformations[6] = partial(poly_deformation, monomials=dm6, coefficients=dc6, eq_idx=basis[0])
	deformations[7] = partial(poly_deformation, monomials=dm7, coefficients=dc7, eq_idx=basis[0])
	deformations[8] = partial(poly_deformation, monomials=dm8, coefficients=dc8, eq_idx=basis[0])

	return deformations

def TY_KM_deformations_expanded(cdtype=np.complex64):
	"""
	Diagonal basis for $H^{(0,1)}(X,TX)$ after Gram-Schmidt using WP
	See Kalara, Mohapatra, Phys. Review D, 1987.
	No Z3 quotient.
	"""
	deformation_monomials = [
		np.asarray([[1, 1, 1, 0, 0, 0, 0, 0]]),  # λ1 - e1 | LEPTONS
		np.asarray([[1, 1, 0, 1, 0, 0, 0, 0]]),  # λ2

		np.asarray([[0, 0, 0, 0, 1, 1, 1, 0]]),  # λ3 - e2
		np.asarray([[0, 0, 0, 0, 1, 1, 0, 1]]),  # λ4

		np.asarray([[0, 1, 0, 0, 0, 1, 0, 0]]),  # λ5 - e3
		np.asarray([[0, 0, 1, 0, 0, 0, 1, 0]]),  # λ6
		np.asarray([[0, 0, 0, 1, 0, 0, 0, 1]]),  # λ7
		np.asarray([[0, 0, 1, 0, 0, 0, 0, 1]]),  # λ8
		np.asarray([[0, 0, 0, 1, 0, 0, 1, 0]]),  # λ9	

		
		np.asarray([[0, 1, 1, 1, 0, 0, 0, 0]]),  # Q1 - e1 | QUARKS
		np.asarray([[0, 0, 0, 0, 1, 0, 1, 1]]),  # Q2 - e2
		np.asarray([[1, 2, 0, 0, 0, 0, 0, 0]]),  # Q3 - e1
		np.asarray([[0, 1, 2, 0, 0, 0, 0, 0]]),  # Q4 - e1
		np.asarray([[0, 1, 0, 2, 0, 0, 0, 0]]),  # Q5 - e1
		np.asarray([[2, 0, 1, 0, 0, 0, 0, 0]]),  # Q6 - e1
		np.asarray([[2, 0, 0, 1, 0, 0, 0, 0]]),  # Q7 - e1

		np.asarray([[1, 0, 1, 1, 0, 0, 0, 0]]),  # Qc1 - e1 | ANTIQUARKS
		np.asarray([[0, 0, 0, 0, 0, 1, 1, 1]]),  # Qc2 - e2
		np.asarray([[2, 1, 0, 0, 0, 0, 0, 0]]),  # Qc3 - e1
		np.asarray([[1, 0, 2, 0, 0, 0, 0, 0]]),  # Qc4 - e1
		np.asarray([[1, 0, 0, 2, 0, 0, 0, 0]]),  # Qc5 - e1
		np.asarray([[0, 2, 1, 0, 0, 0, 0, 0]]),  # Qc6 - e1
		np.asarray([[0, 2, 0, 1, 0, 0, 0, 0]]),  # Qc7 - e1
	]

	h_21 = len(deformation_monomials)

	n_hyper = 3  # number of defining equations

	basis = np.split(np.eye(n_hyper), n_hyper)
	_basis = [np.squeeze(b) for b in basis]

	deformation_idx = np.stack((*(_basis[0],)*2, *(_basis[1],)*2, *(_basis[2],)*5))
	deformation_idx = np.vstack((deformation_idx, np.stack((_basis[0], _basis[1], *(_basis[0],)*5))))
	deformation_idx = np.vstack((deformation_idx, np.stack((_basis[0], _basis[1], *(_basis[0],)*5))))
	deformation_idx = np.expand_dims(deformation_idx, axis=1)
	
	dm = jnp.concatenate(deformation_monomials)
	dc_i =	3 * np.asarray([1.], dtype=cdtype)
	deformations = [partial(poly_deformation, monomials=dm[i], coefficients=dc_i, eq_idx=deformation_idx[i]) \
			for i in range(h_21)]
	
	# deformation_coeffs = 3 * np.eye(h_21, dtype=cdtype)
	# dc = jnp.einsum('...i,ij->...ij', deformation_coeffs, deformation_idx)
	# deformations = [partial(alg_geo.evaluate_poly, monomials=dm, coefficients=c.T) for c in dc]

	# Kalara-Mohapatra diagonal basis
	dm4 = [np.asarray([[0, 0, 3, 0, 0, 0, 0, 0]]),
			np.asarray([[0, 0, 0, 3, 0, 0, 0, 0]])]
	dm4 = jnp.concatenate(dm4)
	dc4 = 3 * np.asarray([1.,1.], dtype=cdtype)  # corresponds to KM deformation direction

	dm5 = dm4
	dc5 = 3 * np.asarray([1.,-1.], dtype=cdtype)

	dm6 = [np.asarray([[0, 3, 0, 0, 0, 0, 0, 0]]),
			np.asarray([[3, 0, 0, 0, 0, 0, 0, 0]])]
	dm6 = jnp.concatenate(dm6)
	dc6 = 3 * np.asarray([1.,-1.], dtype=cdtype)

	dm7 = np.asarray([[0, 0, 1, 2, 0, 0, 0, 0]])
	dc7 = 3 * np.asarray([1.], dtype=cdtype)

	dm8 = np.asarray([[0, 0, 2, 1, 0, 0, 0, 0]])
	dc8 = 3 * np.asarray([1.], dtype=cdtype)

	deformations[4] = partial(poly_deformation, monomials=dm4, coefficients=dc4, eq_idx=basis[0])
	deformations[5] = partial(poly_deformation, monomials=dm5, coefficients=dc5, eq_idx=basis[0])
	deformations[6] = partial(poly_deformation, monomials=dm6, coefficients=dc6, eq_idx=basis[0])
	deformations[7] = partial(poly_deformation, monomials=dm7, coefficients=dc7, eq_idx=basis[0])
	deformations[8] = partial(poly_deformation, monomials=dm8, coefficients=dc8, eq_idx=basis[0])

	return deformations

def tian_yau_KM_yukawas_quarks():
	"""
	Yukawa couplings of type QQcλ. See Table VIII in [Kalara-Mohapatra '87].
	"""
	Q_SHIFT		= 9
	QC_SHIFT	= 16
	kappa_deformation_idx = [
		(Q_SHIFT-1 + 5, QC_SHIFT-1 + 7, 5 - 1), # Q5 Q7c λ5
		(Q_SHIFT-1 + 7, QC_SHIFT-1 + 5, 5 - 1), # Q7 Q5c λ5

		(Q_SHIFT-1 + 5, QC_SHIFT-1 + 7, 6 - 1), # Q5 Q7c λ6
		(Q_SHIFT-1 + 7, QC_SHIFT-1 + 5, 6 - 1), # Q7 Q5c λ6

		(Q_SHIFT-1 + 5, QC_SHIFT-1 + 7, 7 - 1), # Q5 Q7c λ7		0
		(Q_SHIFT-1 + 7, QC_SHIFT-1 + 5, 7 - 1), # Q7 Q5c λ7		1

		(Q_SHIFT-1 + 4, QC_SHIFT-1 + 6, 5 - 1), # Q4 Q6c λ5
		(Q_SHIFT-1 + 6, QC_SHIFT-1 + 4, 5 - 1), # Q6 Q4c λ5

		(Q_SHIFT-1 + 4, QC_SHIFT-1 + 6, 6 - 1), # Q4 Q6c λ6
		(Q_SHIFT-1 + 6, QC_SHIFT-1 + 4, 6 - 1), # Q6 Q4c λ6

		(Q_SHIFT-1 + 4, QC_SHIFT-1 + 6, 7 - 1), # Q4 Q6c λ7		2
		(Q_SHIFT-1 + 6, QC_SHIFT-1 + 4, 7 - 1), # Q6 Q4c λ7		3

		(Q_SHIFT-1 + 2, QC_SHIFT-1 + 7, 1 - 1), # Q2 Q7c λ1
		(Q_SHIFT-1 + 7, QC_SHIFT-1 + 2, 1 - 1), # Q7 Q2c λ1

		(Q_SHIFT-1 + 3, QC_SHIFT-1 + 5, 1 - 1), # Q3 Q5c λ1
		(Q_SHIFT-1 + 5, QC_SHIFT-1 + 3, 1 - 1), # Q5 Q3c λ1

		(Q_SHIFT-1 + 3, QC_SHIFT-1 + 5, 2 - 1), # Q3 Q5c λ2
		(Q_SHIFT-1 + 5, QC_SHIFT-1 + 3, 2 - 1), # Q5 Q3c λ2

		(Q_SHIFT-1 + 1, QC_SHIFT-1 + 2, 5 - 1), # Q1 Q2c λ5
		(Q_SHIFT-1 + 2, QC_SHIFT-1 + 1, 5 - 1), # Q2 Q1c λ5

		(Q_SHIFT-1 + 1, QC_SHIFT-1 + 2, 6 - 1), # Q1 Q2c λ6
		(Q_SHIFT-1 + 2, QC_SHIFT-1 + 1, 6 - 1), # Q2 Q1c λ6

		(Q_SHIFT-1 + 3, QC_SHIFT-1 + 3, 5 - 1), # Q3 Q3c λ5

		(Q_SHIFT-1 + 3, QC_SHIFT-1 + 3, 6 - 1), # Q3 Q3c λ6
	]

	return kappa_deformation_idx
