import jax
import jax.numpy as jnp
from jax import jit, vmap

import math
import numpy as np
import sympy as sp

from functools import partial
from abc import ABC, abstractmethod
from itertools import combinations_with_replacement

from . import math_utils

def monomial_idx(ambient, degree):
    idx = list(combinations_with_replacement(np.arange(ambient.item()+1), degree))
    return np.array(idx, dtype=np.int32)

def mono_construct(A, i):
    A = A.at[i].add(1)
    return A, i

def monomial_idx_to_powers(monomial_idx, ambient):
    m_k = np.zeros((monomial_idx.shape[0], sum(ambient).item() + len(ambient)), dtype=np.int32)
    s_k, _ = vmap(jax.lax.scan, in_axes=(None,0,0))(mono_construct, m_k, monomial_idx)
    return np.array(s_k)

def monomial_basis(ambient, degree):
    _monomial_idx = monomial_idx(ambient, degree)
    return monomial_idx_to_powers(_monomial_idx, ambient)

def reduced_monomial_basis(ambient, degree, monomials_mod):
    # only works for hypersurfaces
    monomials_mod = np.atleast_2d(monomials_mod)
    mod_degree = np.sum(monomials_mod, axis=1)
    assert np.all(mod_degree == np.roll(mod_degree, 1))
    mod_degree = mod_degree[0]
    
    if degree < mod_degree:
        return monomial_basis(ambient, degree)
        
    pows_aux = monomial_basis(ambient, degree - mod_degree)
    pows = [tuple(p) for p in monomial_basis(ambient, degree)]

    for pow_aux in pows_aux:
        mod_element = monomials_mod + pow_aux
        for mono in mod_element:
            try:
                pows.remove(tuple(mono))
                break  # remove one to break linear dependency
            except ValueError:
                continue
    return np.array(pows)

def monomial_basis_size(ambient, degree, constraint=False):
    # constraint determined by defining polynomial
    n, k = ambient.item(), degree
    Pn_k_n_monomials = math.comb(k + n, k)
    if constraint is False:
        return Pn_k_n_monomials
    return Pn_k_n_monomials - math.comb(k-1, k - (n+1))

def evaluate_monomial(points, monomials):
    r"""
    Evaluates polynomial defined by monomials and coefficients
    mono(z) = \prod_i z_i^mono_i
    """
    mono_eval = jnp.prod(jnp.power(points, monomials), axis=-1)
    return mono_eval

@partial(jit, static_argnums=(2,))
def monomial_evaluate_log(p, s_k, conj=False):
    p = math_utils.to_complex(p)
    poly_eval = jnp.exp(jnp.einsum('...i, ...ji->...j', jnp.log(p), s_k))
    if conj is True: return jnp.conjugate(poly_eval)
    return poly_eval

@jit
def poly_evaluate_log(p, s_k, coeffs):
    mono_eval = monomial_evaluate_log(p, s_k)
    return jnp.sum(coeffs * mono_eval, axis=-1)

def monomials_to_power_matrix(monomials, variables):
    """Convert a list of SymPy monomials to a power matrix.
    """
    power_matrix = []
    
    for monomial in monomials:
        powers_dict = monomial.as_powers_dict()
        powers = [powers_dict.get(var, 0) for var in variables]
        power_matrix.append(powers)
    
    return np.asarray(power_matrix, dtype=np.int32)

def powers_to_poly(powers, variables):
    return sp.Mul(*[variables[i]**p for i, p in enumerate(powers) if p != 0])

def get_quotient_basis(variables, monad_map, monomials_B, monomials_C):
    r"""Computes basis of polynomials for $H^1(X;V)$ in the LES
    associated to $$ 0 \rightarrow V \rightarrow B \rightarrow C $$
    """
    from sympy import groebner
    
    # Generate the ideal generators
    # Each generator is a map component multiplied by a variable
    ideal_generators = []
    monad_map_power_matrix = monomials_to_power_matrix(monad_map, variables)
    
    for m in monad_map_power_matrix:
        for m_b in monomials_B:
            ideal_generators.append(powers_to_poly(m + m_b, variables))

    G = groebner(ideal_generators)

    quotient_basis = []
    all_monomials = monomials_C
    for mono in all_monomials:
        if not G.contains(powers_to_poly(mono, variables)):
            quotient_basis.append(mono)
    # e.g.
    # monomial = powers_to_poly([0,0,0,3,0], variables)
    # G.contains(monomial)

    quotient_basis = np.array(quotient_basis)
    quotient_basis = quotient_basis[np.lexsort(np.rot90(quotient_basis))]
    print(f'Dimension of quotient: {quotient_basis.shape}')
    return quotient_basis, ideal_generators, G


class LineBundleSections(ABC):

    def __init__(self, degree, power_matrix):
        self.degree = degree
        self.power_matrix = power_matrix

    @property
    def size(self):
        return self.power_matrix.shape[0]
    
    def __call__(self, z):
        return monomial_evaluate_log(z, self.power_matrix)

class MonomialBasis(LineBundleSections):
    
    def __init__(self, ambient, degree: int):
        """Full set of homogeneous monomials on projective space."""
        self.ambient = ambient
        power_matrix = monomial_basis(ambient, degree)
        super().__init__(degree, power_matrix)

    @property
    def size(self):
        n = self.ambient[0]
        return math.comb(n + self.degree, self.degree)

class MonomialBasisReduced(LineBundleSections):
    # work with hypersurfaces for now
    def __init__(self, ambient, degree: int, defining_polys, psi=None):
        self.ambient = ambient
        def_poly = defining_polys
        self.def_poly = def_poly
        if psi is not None:
            _psi = sp.Symbol('psi')
            def_poly = def_poly.subs(_psi, psi)

        # monomials to quotient out
        monomials_mod = np.array([x[0] for x in sp.Poly(def_poly, domain='CC').terms()], dtype=np.int32)
        mod_degree = np.sum(monomials_mod, axis=1)
        assert np.all(mod_degree == np.roll(mod_degree, 1))
        self.mod_degree = mod_degree[0]
        power_matrix = reduced_monomial_basis(ambient, degree, monomials_mod)
        super().__init__(degree, power_matrix)

    @property
    def size(self):
        n = self.ambient[0]
        # (n + p - 1, p), p = n - deg(P)
        if self.degree < self.mod_degree:
            return math.comb(n + self.degree, self.degree)
        p = self.degree - self.mod_degree
        return math.comb(n + self.degree, self.degree) - math.comb(n + p, p)

def dim_OXk(ambient, degree, mod_degree):
    r"""
    Finds dimension of $O_X(k)$ over algebraic variety given by 
    zero locus of degree `mod_degree`.
    """
    n = ambient[0]
    # (n + p - 1, p), p = n - deg(P)
    if degree < mod_degree:
        return math.comb(n + degree, degree)
    p = degree - mod_degree
    return math.comb(n + degree, degree) - math.comb(n + p, p)


