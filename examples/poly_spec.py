"""
Polynomial data for deformation families considered in https://arxiv.org/abs/2407.13836.
"""

import numpy as np
import jax.numpy as jnp

def tian_yau_spec():

    # At Landau-Ginzburg point.
    # deformation along e0, p0 -> p0 + x0x1x2
    monomials_1 = np.asarray([
        [3, 0, 0, 0, 0, 0, 0, 0],
        [0, 3, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0]], dtype=np.int64)

        # deformation along e0, p0 -> p0 + x0x1x2
        # [1, 1, 1, 0, 0, 0, 0, 0]], dtype=np.int64)

    monomials_2 = np.asarray([
        [0, 0, 0, 0, 3, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 3, 0],
        [0, 0, 0, 0, 0, 0, 0, 3]], dtype=np.int64)

    monomials_3 = np.asarray([
        [1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 1]], dtype=np.int64)
    monomials = [monomials_1, monomials_2, monomials_3]

    cy_dim = 3
    kmoduli = np.ones(2, dtype=np.complex64)
    ambient = np.array([3,3])

    return monomials, cy_dim, kmoduli, ambient


def bicubic_redux_spec():
    monomials = np.asarray([
        [3, 0, 0, 3, 0, 0],
        [0, 3, 0, 0, 3, 0],
        [0, 0, 3, 0, 0, 3],
        [3, 0, 0, 0, 3, 0],
        [0, 3, 0, 0, 0, 3],
        [0, 0, 3, 3, 0, 0],
        [3, 0, 0, 0, 0, 3],
        [0, 3, 0, 3, 0, 0],
        [0, 0, 3, 0, 3, 0],
    ], dtype=np.int64)
    monomials = [monomials]

    cy_dim = 3
    kmoduli = np.ones(2, dtype=np.complex64)
    # kmoduli = np.array([1,0], dtype=np.complex64)
    ambient = np.array([2,2])

    return monomials, cy_dim, kmoduli, ambient

def bicubic_redux_coefficients(psi):
    coefficients = [np.concatenate((np.ones(3), psi * np.ones(6)))]
    return coefficients

def quarti_quadric_spec():
    monomials = np.asarray([
        [4, 0, 0, 0, 2, 0],
        [0, 4, 0, 0, 2, 0],
        [0, 0, 4, 0, 2, 0],
        [0, 0, 0, 4, 2, 0], 
        [4, 0, 0, 0, 0, 2],
        [0, 4, 0, 0, 0, 2], 
        [0, 0, 4, 0, 0, 2],
        [0, 0, 0, 4, 0, 2],
        [1, 1, 1, 1, 1, 1],
    ], dtype=np.int64)
    monomials = [monomials]
    
    cy_dim = 3
    # kmoduli = 12 * np.ones(2, dtype=np.complex64)
    t0, t1 = 12, 6
    kmoduli = np.array([t0,t1], dtype=np.complex64)
    ambient = np.array([3,1])

    return monomials, cy_dim, kmoduli, ambient

def quarti_quadric_coefficients(psi):
    coefficients = [np.asarray([1, 1, 1, 1, 1, 2, -1, -2, -4 * psi])]
    return coefficients

def tian_yau_coefficients(psi):
    coefficients = [np.append(np.ones(4), -3. * psi), np.ones(4), np.ones(4)]
    return coefficients

def tian_yau_KM_spec():
    # Based off deformations in Kalara-Mohapatra paper
    # At Landau-Ginzburg point.
    monomials_1 = np.asarray([
        [3, 0, 0, 0, 0, 0, 0, 0],
        [0, 3, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 0, 0]], dtype=np.int64)

    monomials_2 = np.asarray([
        [0, 0, 0, 0, 3, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 3, 0],
        [0, 0, 0, 0, 0, 0, 0, 3]], dtype=np.int64)

    monomials_3 = np.asarray([
        [1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 1]], dtype=np.int64)
    monomials = [monomials_1, monomials_2, monomials_3]

    cy_dim = 3
    kmoduli = np.ones(2)
    ambient = np.array([3,3])

    return monomials, cy_dim, kmoduli, ambient

def tian_yau_KM_coefficients(psi):
    coefficients = [np.ones(4), np.ones(4), np.append(np.ones(2), (1. + psi) * np.ones(2))]
    return coefficients

def schimmrigk_spec():

    monomials_1 = np.asarray([
        [3, 0, 0, 0, 0, 0, 0],
        [0, 3, 0, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 0]], dtype=np.int64)

    monomials_2 = np.asarray([
        [0, 1, 0, 0, 3, 0, 0],
        [0, 0, 1, 0, 0, 3, 0],
        [0, 0, 0, 1, 0, 0, 3]], dtype=np.int64)

    monomials = [monomials_1, monomials_2]

    cy_dim = 3
    kmoduli = np.ones(2, dtype=np.complex64)
    ambient = np.array([3,2])
    return monomials, cy_dim, kmoduli, ambient

def schimmrigk_coefficients(psi):
    coefficients = [np.ones(4), np.ones(3)]
    return coefficients

"""
Single parameter families.
"""

def X33_spec():

    monomials_1 = np.asarray([
        [3, 0, 0, 0, 0, 0],
        [0, 3, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 0],
        [0, 0, 0, 1, 1, 1]], dtype=np.int64)

    monomials_2 = np.asarray([
        [0, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 3, 0],
        [0, 0, 0, 0, 0, 3],
        [1, 1, 1, 0, 0, 0]], dtype=np.int64)

    monomials = [monomials_1, monomials_2]

    cy_dim = 3
    kmoduli = np.ones(1, dtype=np.complex64)
    ambient = np.array([5])
    return monomials, cy_dim, kmoduli, ambient

def X33_coefficients(psi):
    coefficients = [np.append(np.ones(3), -3.0*psi), np.append(np.ones(3), -3.0*psi)]
    return coefficients

def X24_spec():
    monomials_1 = np.asarray([
        [2, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 1, 1]], dtype=np.int64)

    monomials_2 = np.asarray([
        [0, 0, 0, 0, 4, 0],
        [0, 0, 0, 0, 0, 4],
        [1, 1, 1, 1, 0, 0]], dtype=np.int64)

    monomials = [monomials_1, monomials_2]

    cy_dim = 3
    kmoduli = np.ones(1, dtype=np.complex64)
    ambient = np.array([5])
    return monomials, cy_dim, kmoduli, ambient  

def X24_coefficients(psi):
    coefficients = [np.append(np.ones(4), -2.0*psi), np.append(np.ones(2), -4.0*psi)]
    return coefficients

def X223_spec():

    monomials_1 = np.asarray([
        [2, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1]], dtype=np.int64)

    monomials_2 = np.asarray([
        [0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0],
        [1, 1, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    monomials_3 = np.asarray([
        [0, 0, 0, 0, 0, 3, 0],
        [0, 0, 0, 0, 0, 0, 3],
        [0, 0, 1, 1, 1, 0, 0]], dtype=np.int64)

    monomials = [monomials_1, monomials_2, monomials_3]

    cy_dim = 3
    kmoduli = np.ones(1, dtype=np.complex64)
    ambient = np.array([6])
    return monomials, cy_dim, kmoduli, ambient  

def X223_coefficients(psi):
    coefficients = [np.append(np.ones(3), -2.0*psi), np.append(np.ones(2), -2.0*psi),
                    np.append(np.ones(2), -3.0*psi)]
    return coefficients

def X2222_spec():
    monomials_1 = np.asarray([
        [2, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0]], dtype=np.int64)
    
    monomials_2 = np.asarray([
        [0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0]], dtype=np.int64)
    
    monomials_3 = np.asarray([
        [0, 0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1]], dtype=np.int64)
    
    monomials_4 = np.asarray([
        [0, 0, 0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 2],
        [1, 1, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

    monomials = [monomials_1, monomials_2, monomials_3, monomials_4]

    cy_dim = 3
    kmoduli = np.ones(1, dtype=np.complex64)
    ambient = np.array([7])
    return monomials, cy_dim, kmoduli, ambient  

def X2222_coefficients(psi):
    coefficients = [np.append(np.ones(2), -2.0*psi), np.append(np.ones(2), -2.0*psi),
                    np.append(np.ones(2), -2.0*psi), np.append(np.ones(2), -2.0*psi)]
    return coefficients

def mirror_quintic_spec():
    monomials = np.asarray([
        [5, 0, 0, 0, 0], # z0^5
        [0, 5, 0, 0, 0], # z1^5
        [0, 0, 5, 0, 0], # z2^5
        [0, 0, 0, 5, 0], # z3^5
        [0, 0, 0, 0, 5], # z4^5

        [1, 1, 1, 1, 1],
    ], dtype=np.int64)

    cy_dim, n_coords = 3, monomials.shape[-1]
    kmoduli = np.ones(1, dtype=np.complex64)
    ambient = np.array([4])
    degrees = ambient + 1

    return monomials, cy_dim, kmoduli, ambient

def mirror_quintic_coefficients(psi):
    coefficients = np.append(np.ones(5), -5. * psi)
    return coefficients

def mirror_quintic_deformation(p, precision=np.complex128):
    mirror_quintic_deform_monomial = jnp.asarray([1, 1, 1, 1, 1], dtype=np.int32)
    deformation_monomial = jnp.prod(jnp.power(p, mirror_quintic_deform_monomial), axis=-1)
    return jnp.einsum('...a, aj->...j', jnp.expand_dims(deformation_monomial, axis=-1),
                   jnp.asarray([[-5.]], precision))

def X33_deformation(p, precision=np.complex128):
    d1 = jnp.einsum("...a,aj->...j", jnp.expand_dims(p[3]*p[4]*p[5], axis=-1),
                      jnp.asarray([[-3.,0.]], precision))
    d2 = jnp.einsum("...a,aj->...j", jnp.expand_dims(p[0]*p[1]*p[2], axis=-1),
                      jnp.asarray([[0.,-3.]], precision))
    return d1 + d2

def tian_yau_deformation(p, precision=np.complex128):
    # example tangent vector in M_{cs} for testing. NB: this is one of 9.
    return jnp.einsum("...a,aj->...j", jnp.expand_dims(p[0]*p[1]*p[2], axis=-1), 
                      jnp.asarray([[-3.,0,0]], precision))

def tian_yau_KM_deformation(p, precision=np.complex128):
    # example tangent vector in M_{cs} for testing. NB: this is one of 9.
    d1 = jnp.einsum("...a,aj->...j", jnp.expand_dims(p[2]**3+p[3]**3, axis=-1), 
                      jnp.asarray([[0,0,1.]], precision))
    return d1

def X2222_deformation(z_pts, precision=np.complex128):
    d1 = jnp.einsum("...a,aj->...j", jnp.expand_dims(z_pts[2]*z_pts[3], axis=-1),
                      jnp.asarray([[2.,0.,0.,0.]], precision))
    d2 = jnp.einsum("...a,aj->...j", jnp.expand_dims(z_pts[4]*z_pts[5], axis=-1),
                      jnp.asarray([[0.,2.,0.,0.]], precision))
    d3 = jnp.einsum("...a,aj->...j", jnp.expand_dims(z_pts[6]*z_pts[7], axis=-1),
                      jnp.asarray([[0.,0.,2.,0.]], precision))
    d4 = jnp.einsum("...a,aj->...j", jnp.expand_dims(z_pts[0]*z_pts[1], axis=-1),
                      jnp.asarray([[0.,0.,0.,2.]], precision))
    return d1 + d2 + d3 + d4

def X24_deformation(p, precision=np.complex128):
    d1 = jnp.einsum("...a,aj->...j", jnp.expand_dims(p[4]*p[5], axis=-1),
                      jnp.asarray([[-2.,0.]], precision))
    d2 = jnp.einsum("...a,aj->...j", jnp.expand_dims(p[0]*p[1]*p[2]*p[3], axis=-1),
                      jnp.asarray([[0.,-4.]], precision))
    return d1 + d2

def quarti_quadric_deformation(p, precision=np.complex128):
    d = jnp.expand_dims(jnp.prod(p, dtype=precision), axis=-1)
    return d

def tian_yau_yukawas():
    kappa_deformation_idx = [
        (0,2,4),
        (0,2,5),
        (0,2,6),
        (1,3,4),
        (1,3,5),
        (1,3,6),
        (0,3,8),
        (1,2,7),
        (5,5,4),
        (6,6,4),
        (6,6,5),
        (4,5,6)
    ]
    return kappa_deformation_idx

def tian_yau_KM_yukawas():
    kappa_deformation_idx = [
        (0,2,4),
        (0,2,5),
        (0,2,6),
        (1,3,4),
        (1,3,5),
        (1,3,6),
        (0,3,8),
        (1,2,7),
        (4,4,5),
        (4,4,6),
        (5,5,4),
        (5,5,6),
        (6,6,4),
        (6,6,5),
        (4,5,6),
        (7,8,4),
        (7,8,5),
        (7,8,6)

    ]
    return kappa_deformation_idx

"""
Jax versions of coefficients for autodiff
"""

def _mirror_quintic_coefficients(psi):
    coefficients = jnp.append(jnp.ones(5), -5. * psi)
    return coefficients

def _tian_yau_coefficients(psi):
    coefficients = [jnp.append(jnp.ones(4), -3. * psi), jnp.ones(4), jnp.ones(4)]
    return coefficients

def _tian_yau_KM_coefficients(psi):
    coefficients = [jnp.ones(4), jnp.ones(4), jnp.append(jnp.ones(2), (1. + psi) * jnp.ones(2))]
    return coefficients

def _schimmrigk_coefficients(psi):
    coefficients = [jnp.ones(4), jnp.ones(3)]
    return coefficients

def _X33_coefficients(psi):
    coefficients = [jnp.append(jnp.ones(3), -3.0*psi), jnp.append(jnp.ones(3), -3.0*psi)]
    return coefficients

def _X24_coefficients(psi):
    coefficients = [jnp.append(jnp.ones(4), -4.0*psi), jnp.append(jnp.ones(2), -2.0*psi)]
    return coefficients

def _X223_coefficients(psi):
    coefficients = [jnp.append(jnp.ones(3), -3.0*psi), jnp.append(jnp.ones(2), -2.0*psi),
                    jnp.append(jnp.ones(2), -2.0*psi)]
    return coefficients

def _X2222_coefficients(psi):
    coefficients = [jnp.append(jnp.ones(2), -2.0*psi), jnp.append(jnp.ones(2), -2.0*psi),
                    jnp.append(jnp.ones(2), -2.0*psi), jnp.append(jnp.ones(2), -2.0*psi)]
    return coefficients
