r"""
Model architectures for approximations of tensor fields on manifolds. The idea behind 
the form of each ansatz is that:

1. The tensor field is parameterised using a single (possibly vector-valued) function, $\phi \in C^{\infty}(X)$.
2. The resulting tensor field should be globally defined over the manifold.

(2.) basically means that the local representation of the tensor field in each coordinate chart should glue 
together coherently. This particularly important for manifolds with a nontrivial topology - one appears to obtain
nonsensical answers for downstream predictions if this is not respected.

The `flax` library is used here to define the models, but this is interchangable with any library or
framework. As idiomatic jax is functional, the model definitions are kept separate from the parameters. 
During the forward pass, the parameters are passed separately as a dictionary to the `apply` method of 
the model.

There are multiple routes to add new architectures for approximation of various tensor fields. The simplest 
one, keeping in line with the logic in the models module, is to add:

1. A [Flax module](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html) describing 
the sequence of operations defined by your architecture.
```py
import jax.numpy as jnp
from flax import linen as nn

class MyAnsatz(nn.Module):

@nn.compact
def __call__(self, local_coords):
    p = local_coords
    p_bar = jnp.conjugate(p)
    p_norm_sq = jnp.sum(p * p_bar)
    return jnp.outer(p, p_bar) / p_norm_sq
```
2. A pure function which accepts a pytree of parameters for the model and executes the computation by 
invoking the `.apply` method of the module you defined above.
```py
def ansatz_fun(p, params, *args):
    p = ...  # some logic
    model = MyAnsatz(*args)  # model constructor
    return model.apply({'params': params}, p)
```
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, jacfwd

import numpy as np

from flax import linen as nn

from functools import partial
from typing import Callable, Sequence, Mapping
from jaxtyping import Array, Float, Complex, ArrayLike

# custom
from .. import alg_geo, curvature, fubini_study
from ..utils import math_utils, ops

class LearnedVector_spectral_nn(nn.Module):
    r"""
    Spectral network implementation for hypersurfaces embedded in $\mathbb{P}^n$.
    The model defined here is a simple feedforward network with a spectral embedding
    layer at the input, which converts points expressed in homogeneous coordinates to
    a $\mathbb{C}^*$-invariant matrix representation. The resulting parameterised function 
    is globally defined over $X$ - one may then construct various ansatze off this.
    For details see [arxiv:2211.09801](https://arxiv.org/abs/2211.09801).

    Attributes
    ----------
    dim : int
        Dimension of projective space + 1.
    ambient : Sequence[int]
        Dimensions of the ambient space factors.
    n_units : Sequence[int], optional
        Number of units in each layer, by default (64, 64, 64).
    n_out : int, optional
        Number of output features, by default 1.
    use_spectral_embedding : bool, optional
        Whether to use spectral embedding, default True.
    activation : Callable[[jnp.ndarray], jnp.ndarray], optional
        Nonlinearity between each layer, default nn.gelu.
    cy_dim : int, optional
        Dimension of the complex projective space, default 3.
    """
    dim: int
    ambient: Sequence[int]
    n_units: Sequence[int] = (64,64,64)
    n_out: int  = 1
    use_spectral_embedding: bool = True
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    cy_dim: int = 3
    
    def setup(self):
        self.n_hidden = len(self.n_units)
        self.layers = [nn.Dense(f) for f in self.n_units]

    def spectral_layer(self, x: Float[Array, "i"], x_dim: int) -> Array:
        r"""Converts homogeneous $[z_0 : ... : z_n]$ coords in $\mathbb{P}^n$ 
        into the first-order basis of eigenfunctions of the Laplacian in
        projective space, whose generic form is $z_i \bar{z}_j / \sum z_k \bar{z}_k$.

        Parameters
        ----------
        x : Float[Array, "i"]
            Input array of homogeneous coordinates.
        x_dim : int
            Dimension of the input array.

        Returns
        -------
        Complex[Array, "i * (i+1) // 2"]
            First-order basis of eigenfunctions of the Laplacian as a vector.
        """
        print(f"Compiling {self.spectral_layer.__qualname__}.")
        p =  math_utils.to_complex(x)
        p_bar = jnp.conjugate(p)
        p_norm_sq = jnp.sum(p * p_bar)
        alpha = jnp.outer(p, p_bar) / p_norm_sq
        alpha_reduced = alpha[jnp.triu_indices(x_dim)]

        surviving = math_utils.to_real(alpha_reduced)
        return surviving[jnp.nonzero(surviving, size=int(x_dim**2))]

    def _spectral_layer_real_ops(self, z, z_dim):
        z = jnp.squeeze(z)
        x, y = jnp.split(z, 2)  # assuming real input divided as [Re(z) | Im(z)]
        z_norm = jnp.sum(z**2)
        Re_alpha, Im_alpha = jax.vmap(math_utils.complex_mult, in_axes=(0,0,None,None))(x,y,x,y)
        Re_alpha = Re_alpha[jnp.triu_indices(z_dim)]
        Im_alpha = Im_alpha[jnp.triu_indices(z_dim)]
        Im_alpha = Im_alpha[jnp.nonzero(Im_alpha, size=z_dim*(z_dim-1)//2)]
        return jnp.concatenate([Re_alpha, Im_alpha])/z_norm

    @nn.compact
    def __call__(self, x: Float[Array, "i"]) -> Array:
        """Spectral NN forward pass for hypersurfaces embedded in a single
        projective space factor.

        Parameters
        ----------
        x : Float[Array, "i"]
            Input array of homogeneous coordinates.

        Returns
        -------
        Complex[Array, "n_out"]
            Output of vector-valued function.
        """
        if self.use_spectral_embedding is True:
            x = self.spectral_layer(x, self.dim)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != self.n_hidden - 1:
                x = self.activation(x)
            
        out = nn.Dense(self.n_out, name='scalar')(x)
        return jnp.squeeze(out)

class LearnedVector_spectral_nn_CICY(LearnedVector_spectral_nn):
    r"""Spectral network implementation for manifolds embedded in a product of projective
    space factors, ${P}^{n_1} \times \cdots \times \mathbb{P}^{n_K}$.
    """
    def setup(self):
        self.n_hidden = len(self.n_units)
        self.layers = [nn.Dense(f) for f in self.n_units]
        self.dims = np.array(self.ambient) + 1

    @nn.compact
    def __call__(self, x: Float[Array, "i"]) -> Array:
        r"""Spectral NN forward pass for complete intersection manifolds in a product
        of projective spaces. The spectral layer converts the coordinates for each 
        projective space to its respective $\mathbb{C}^*$-invariant matrix representation,
        and concatenates the results.

        Parameters
        ----------
        x : Float[Array, "i"]
            Input array of homogeneous coordinates, local coordinates for each
            projective space factor are listed consecutively in the input array.

        Returns
        -------
        Complex[Array, "n_out"]
            Output of vector-valued function.
        """        
        if self.use_spectral_embedding is True:
            x = math_utils.to_complex(jnp.squeeze(x))
            spectral_out = []

            for i in range(len(self.ambient)):
                s, e = int(np.sum(self.ambient[:i]) + i), int(np.sum(self.ambient[:i+1]) + i + 1)
                p_ambient_i = jax.lax.dynamic_slice(x, (s,), (e-s,))
                p_ambient_i = math_utils.to_real(p_ambient_i)
                spectral_out.append(self.spectral_layer(p_ambient_i, self.dims[i]))

            x = jnp.stack(spectral_out, axis=-1).reshape(-1)
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != self.n_hidden - 1:
                x = self.activation(x)
            
        out = nn.Dense(self.n_out, name='scalar')(x)
        return jnp.squeeze(out)
    

class CoeffNetwork_spectral_nn_CICY(LearnedVector_spectral_nn):
    r"""
    Spectral network parameterising the coefficients of a linear combination of a basis
    of sections for the holomorphic bundle $V \rightarrow X$. The sections are globally defined 
    by construction, hence the linear combination is a global section of $V$, parameterised by
    a neural network. Schematically,

    $$
    s:= \sum_I \psi_I^{(\textsf{NN})} \cdot \mathbf{e}^I~.
    $$

    Where summation over the multi-index $I$ denotes contraction of appropriate tensor indices.
    Here the network specialises to the case of $V=T_X$, the standard embedding, but interchangeable
    with any other bundle $V \rightarrow X$ by subclassing and modifying the shapes of the coefficients.

    The case of multiple parameterised sections $s^{(k)} \in \Gamma(V)$ modelled is handled 
    by the `h_21` parameter - this sets the number of coefficients output by the network.

    Inherits from `LearnedVector_spectral_nn`.

    Attributes
    ----------
    h_21 : int
        Dimension of complex structure moduli space - controls the number of sections learnt.
    """
    n_out: int = -1
    use_spectral_embedding: bool = True
    h_21: int = 1

    def setup(self):
        self.n_hidden = len(self.n_units)
        self.dims = np.array(self.ambient) + 1  # coords for each ambient space factor

        self.param_counts = [(n_c*(n_c-1)//2, (n_c+1)*n_c//2) for n_c in self.dims] 
        self.n_asym, self.n_sym = self.param_counts[-1]
        self.n_eta_out = len(self.ambient) * self.n_asym * self.n_sym * 2
        self.layers = [nn.Dense(f) for f in self.n_units]
        # einsum layers for each ambient space factor. LHS: input, RHS: learnable kernel.
        self.coeff_layer = ops.EinsumComplex((self.n_units[-1], len(self.ambient) * self.h_21, self.n_asym, self.n_sym), 
                                             '...i, ...ihab->...hab', name='layers_coeffs')

    @nn.compact
    def __call__(self, x: Float[Array, "i"]) -> Array:
        """Forward pass for coefficients, modelled as vector-valued functions on $X$. 

        Parameters
        ----------
        x : Float[Array, "i"]
            Input array of homogeneous coordinates, local coordinates for each
            projective space factor are listed consecutively in the input array.

        Returns
        -------
        Complex[Array, "n_out"]
            Output of vector-valued function.
        """      
        # Coefficients and basis
        spectral_out = []
        x = math_utils.to_complex(jnp.squeeze(x))

        for i in range(len(self.ambient)):
            s, e = int(np.sum(self.ambient[:i]) + i), int(np.sum(self.ambient[:i+1]) + i + 1)
            p_ambient_i = jax.lax.dynamic_slice(x, (s,), (e-s,))
            spectral_out.append(self.spectral_layer(math_utils.to_real(p_ambient_i), self.dims[i]))

        x = jnp.stack(spectral_out, axis=-1).reshape(-1)     

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != self.n_hidden - 1:
                x = self.activation(x)

        coeffs = self.coeff_layer(x)  # [..., n_A * h_{21}, n_asym, n_sym]

        print(f'{self.__call__.__qualname__}, coeff shape, {coeffs.shape}')
        return jnp.split(coeffs, len(self.ambient), axis=0)


@partial(jit, static_argnums=(2,3,4,5,6))
def phi_head(p: Float[Array, "i"], params: Mapping[str, Array], n_hyper: int, 
             ambient: Sequence[int], n_out: int = 1, spectral: bool = True, 
             activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu) -> Complex[Array, "n_out"]:
    r"""Wrapper to feed parameters into forward pass for $\phi$-component in 
    the `ddbar_phi_model`.

    Parameters
    ----------
    p : Float[Array, "i"]
        Input array of homogeneous coordinates.
    params : Mapping[Str, Array]
        Model parameters stored as a dictionary - keys are the module names
        registered upon initialisation and values are the parameter values.
    n_hyper : int
        Number of defining equations of the complete intersection.
    ambient : Sequence[int]
        Sequence representing the ambient space dimensions.
    n_out : int, optional
        Dimension of output vector, default 1.
    spectral : bool, optional
        Toggles spectral embedding, default True.
    activation : Callable[[jnp.ndarray], jnp.ndarray], optional
        Activation function, default nn.gelu.

    Returns
    -------
    Complex[Array, "n_out"]
        Output of vector-valued function.
    """
    print(f'Compiling {phi_head.__qualname__}')
    n_units = [params[k]['kernel'].shape[-1] for k in params.keys()][:-1]

    if n_hyper > 1:
        return LearnedVector_spectral_nn_CICY(p.shape[-1]//2, ambient, n_units, n_out, 
                use_spectral_embedding=spectral, activation=activation).apply({'params': params}, p)

    return LearnedVector_spectral_nn(p.shape[-1]//2, ambient, n_units, n_out,
            use_spectral_embedding=spectral, activation=activation).apply({'params': params}, p)

@partial(jit, static_argnums=(2,3))
def ddbar_phi_model(p: Float[Array, "i"], params: Mapping[str, Array], 
                    g_ref_fn: Callable[[ArrayLike], jnp.ndarray], 
                    g_correction_fn: Callable[[ArrayLike], jnp.ndarray]) -> jnp.ndarray:    
    r"""Returns metric on $X$ under pullback from ambient space as an $\partial \bar{\partial}$-exact
    correction to the reference Fubini-Study metric in the desired Kähler class,
    $$ 
    \tilde{g} := g_{\text{ref}} + \partial \overline{\partial} \phi; \quad \phi \in C^{\infty}(X)~.
    $$
    Generates pullbacks on the fly. 

    Parameters
    ----------
    p : Float[Array, "i"]
        Input array of homogeneous coordinates.
    params : Mapping[Str, Array]
        Model parameters stored as a dictionary - keys are the module names
        registered upon initialisation and values are the parameter values.
    g_ref_fn : Callable[[jnp.ndarray], jnp.ndarray]
        Function representing reference Fubini-Study metric in local coordinates.
    g_correction_fn : Callable[[jnp.ndarray], jnp.ndarray]
        Function $\phi \in C^{\infty}(X)$ which generates the $\partial \bar{\partial}$-exact 
        correction to the reference metric.
    """
    print(f'Compiling {ddbar_phi_model.__qualname__}')
    # compute reference form defining Kahler class (in ambient space)
    g_ref_ambient, pullbacks = g_ref_fn(p)

    g_correction = curvature.del_z_bar_del_z(p, g_correction_fn, params, wide=True)
    g_correction = jnp.squeeze(g_correction)
    g_pred_ambient = g_ref_ambient + g_correction

    g_pred = jnp.einsum('...ia,...ab,...jb->...ij', pullbacks, g_pred_ambient,
        jnp.conjugate(pullbacks))

    return g_pred

@partial(jit, static_argnums=(2,3,4,5))
def coeff_head(p: Float[Array, "i"], params: Mapping[str, Array], n_homo_coords: int, 
               ambient: Sequence[int], h_21: int = 1, 
               activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu) -> jnp.ndarray:
    r"""Wrapper to feed parameters into forward pass for section coefficient network.

    Parameters
    ----------
    p : Float[Array, "i"]
        Input array of homogeneous coordinates.
    params : Mapping[Str, Array]
        Model parameters stored as a dictionary - keys are the module names
        registered upon initialisation and values are the parameter values.
    n_homo_coords : int
        Number of homogeneous coordinates.
    ambient : Sequence[int]
        Sequence representing the ambient space dimensions.
    h_21 : int, optional
        Number of sections learnt / harmonic $(0,1)$ forms on $X$. This controls
        the size of the coefficient network.
    activation : Callable[[jnp.ndarray], jnp.ndarray], optional
        Activation function, default nn.gelu.
    """
    print(f'Compiling {coeff_head.__qualname__}')
    # last layer is coeff_layer
    print(sorted(params.keys()))
    n_units = [params[k]['kernel'].shape[-1] for k in sorted(params.keys()) if 'kernel' in params[k].keys()]

    return CoeffNetwork_spectral_nn_CICY(n_homo_coords, ambient, n_units, h_21=h_21,
            activation=activation).apply({'params': params}, p)


def helper_fns(config):
    # Apply partial closure to commonly used functions.
    if config.n_hyper > 1:
        g_FS_fn = partial(fubini_study._fubini_study_metric_homo_gen_pb_cicy, 
                    dQdz_monomials=config.dQdz_monomials, 
                    dQdz_coeffs=config.dQdz_coeffs, 
                    n_hyper=config.n_hyper, 
                    cy_dim=config.cy_dim, 
                    n_coords=config.n_coords,
                    ambient=tuple(config.ambient), 
                    k_moduli=None,
                    ambient_out=True,
                    cdtype=config.cdtype)
        pb_fn = partial(alg_geo._pullbacks_cicy,
                    dQdz_monomials=config.dQdz_monomials, 
                    dQdz_coeffs=config.dQdz_coeffs, 
                    n_hyper=config.n_hyper, 
                    cy_dim=config.cy_dim, 
                    n_coords=config.n_coords,
                    aux=False,
                    cdtype=config.cdtype)
    else:
        g_FS_fn = partial(fubini_study.fubini_study_metric_homo_pb, 
                    dQdz_info=(config.dQdz_monomials, config.dQdz_coeffs),
                    cy_dim=config.cy_dim,
                    ambient_out=True, cdtype=config.cdtype)
        pb_fn = partial(alg_geo.compute_pullbacks,
                    dQdz_info=(config.dQdz_monomials, config.dQdz_coeffs),
                    cy_dim=config.cy_dim, cdtype=config.cdtype)

    g_correction_fn = partial(phi_head, n_hyper=config.n_hyper, ambient=tuple(config.ambient))

    return g_FS_fn, g_correction_fn, pb_fn
