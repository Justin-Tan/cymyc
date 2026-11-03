import jax
import jax.numpy as jnp
from jax import jit, vmap

import math
import numpy as np
import sympy as sp

from functools import partial
from abc import ABC, abstractmethod
from itertools import combinations_with_replacement, product

from . import math_utils

PRIME = 2_147_483_629   # prime < 2^31 with p = 1 mod 4
SQRT_M1 = 629_208_553   # sqrt(-1) mod PRIME, for Gaussian-rational coefficients

def monomial_idx(ambient, degree):
    idx = list(combinations_with_replacement(np.arange(ambient.item()+1), degree))
    return np.array(idx, dtype=np.int32)

def mono_construct(A, i):
    A = A.at[i].add(1)
    return A, i

def monomial_idx_to_powers(monomial_idx, ambient):
    m_k = np.zeros((monomial_idx.shape[0], np.sum(ambient).item() + len(ambient.shape)), dtype=np.int32)
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

def _to_modp(c):
    c = sp.nsimplify(c, rational=True)
    re, im = sp.Rational(sp.re(c)), sp.Rational(sp.im(c))
    return int((re.p * pow(re.q, -1, PRIME) + SQRT_M1 * im.p * pow(im.q, -1, PRIME)) % PRIME)

def groebner_leading_monomials(polys, variables, psi=None, order='grevlex', seed=0):
    r"""Leading monomials of a Groebner basis of $I = (P_1, \dots, P_r)$, i.e. generators of in(I).
    Computed exactly mod a large prime. psi=None -> random (generic) psi: valid for all but
    finitely many psi. Pass special values (e.g. psi=0) explicitly."""
    _psi = sp.Symbol('psi')
    if psi is None:
        psi = int(np.random.default_rng(seed).integers(1, PRIME))
    psi = sp.nsimplify(psi if isinstance(psi, (int, sp.Basic)) else complex(psi), rational=True)
    gf = []
    for P in polys:
        terms = sp.Poly(sp.expand(P.subs(_psi, psi)), *variables).terms()
        coeffs = {m: _to_modp(c) for m, c in terms}
        gf.append(sp.Poly.from_dict({m: c for m, c in coeffs.items() if c}, *variables, modulus=PRIME))
    G = sp.groebner(gf, *variables, order=order, modulus=PRIME, method='f5b')
    return np.array([g.monoms(order=order)[0] for g in G.polys], dtype=np.int32)

def standard_monomial_basis(ambient, degree, lead_monos):
    r"""Degree-k monomials not divisible by any leading monomial of a Groebner basis: a basis of
    $S_k / I_k = H^0(X, O_X(k))$ for a complete intersection X of dim >= 1 (Macaulay)."""
    pows = monomial_basis(ambient, degree)
    lead_monos = np.atleast_2d(lead_monos)
    divisible = np.any(np.all(pows[:, None, :] >= lead_monos[None, :, :], axis=-1), axis=1)
    return pows[~divisible]

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

class MonomialBasisReducedHypersurface(LineBundleSections):
    # work with hypersurfaces for now
    def __init__(self, ambient, degree: int, defining_polys, psi=None):
        self.ambient = ambient
        self.def_polys = defining_polys
        if psi is not None:
            _psi = sp.Symbol('psi')
            self.def_polys = [poly.subs(_psi, psi) for poly in defining_polys]

        # TODO: extend to CICYs
        # monomials to quotient out
        monomials_mod = np.array([x[0] for x in sp.Poly(self.def_polys[0], domain='CC').terms()], dtype=np.int32)
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

class MonomialBasisReduced(LineBundleSections):
    # complete intersections in a single P^n
    def __init__(self, ambient, degree: int, defining_polys, psi=None, variables=None, order='grevlex'):
        self.ambient = ambient
        self.def_polys = defining_polys
        _psi = sp.Symbol('psi')
        if psi is not None:
            self.def_polys = [poly.subs(_psi, psi) for poly in defining_polys]

        if variables is None:  # natural order z_0 < z_1 < ... < z_10; must match column order of points
            syms = set().union(*[P.free_symbols for P in defining_polys]) - {_psi}
            variables = sp.Poly(sp.Add(*syms)).gens
        self.variables = variables
        self.mod_degrees = [sp.Poly(P, *variables).total_degree() for P in defining_polys]
        self.mod_degree = self.mod_degrees[0]  # legacy attribute, only meaningful for hypersurfaces

        self.lead_monos = groebner_leading_monomials(defining_polys, variables, psi=psi, order=order)
        power_matrix = standard_monomial_basis(ambient, degree, self.lead_monos)
        assert power_matrix.shape[0] == dim_OXk(ambient, degree, self.mod_degrees), \
            "defining polynomials are not a complete intersection"
        super().__init__(degree, power_matrix)

    @property
    def size(self):
        return dim_OXk(self.ambient, self.degree, self.mod_degrees)

def dim_OXk_hypersurface(ambient, degree, mod_degree):
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

def dim_OXk(ambient, degree, mod_degree):
    r"""
    Dimension of $H^0(X, O_X(k))$ for a complete intersection X in P^n of dim >= 1, with
    defining degrees `mod_degree` (int for a hypersurface, or a list), from the Koszul complex:
    $\sum_{J} (-1)^{|J|} \binom{n + k - d_J}{n}$, $d_J = \sum_{j \in J} d_j$.
    """
    n = int(np.asarray(ambient).ravel()[0])
    mod_degrees = np.atleast_1d(mod_degree)
    dim = 0
    for J in product((0, 1), repeat=len(mod_degrees)):
        m = degree - int(np.dot(J, mod_degrees))
        if m >= 0:
            dim += (-1) ** sum(J) * math.comb(n + m, n)
    return dim

