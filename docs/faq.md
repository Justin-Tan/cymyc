---
icon: material/help-circle-outline
---
# FAQ
Questions you may or may not have had about this library. 

### Why write in it `Jax`?
`Jax` is a high-performance Python library for program transformations - chief among these being automatic differentiation. This is the transformation of a program into another another,
$$
\texttt{program} \rightarrow \partial(\texttt{program})~,
$$ 
which evaluates the partial derivative with respect to any of $\texttt{program}$'s original inputs. From a computational geometry perspective, this is a boon for computation of curvature-related quantities and differential operators on manifolds. For example, given a program which outputs the metric tensor $g_{\mu \nu}$ in local coordinates, one schematically arrives at the various horsemen of curvature via derivatives w.r.t. local coordinates,

$$ 
\left(g_{\mu \nu} \sim \partial \partial \varphi \right) \stackrel{\partial}{\rightarrow} \left(\Gamma^{\kappa}_{\mu \nu} \sim g \cdot \partial g\right) \stackrel{\partial}{\rightarrow} \left(R^{\kappa}_{\, \, \lambda \mu \nu} \sim \partial \Gamma + \Gamma \cdot \Gamma\right) \rightarrow \cdots ~.
$$

What distinguishes `Jax` from other autodiff / machine learning frameworks is that *idiomatic `Jax` uses a functional programming paradigm*. The price one pays for the significant performance boost afforded by `Jax` for most scientific computing applications are additional constraints on program logic, which would not be present in Python or other libraries which use an imperative paradigm. 

Somewhat loosely, when using `Jax`, one is usually not writing code to be executed by the Python interpreter, rather building a graph of computations which will be compiled and passed to an accelerator, which is typically orders of magnitude faster than regular Python code (and $\mathcal{O}(1)$ faster than Torch/Tensorflow, in our experience). The flip side is that the compilation procedure restricts the program logic to a subset of possible operations relative to other autodiff frameworks.

A full discussion of the `Jax` model is beyond the scope here, and we defer to the excellent [official guides](https://jax.readthedocs.io/en/latest/key-concepts.html) on this matter. However, as a quick summary:

1. The `Jax` computational model is to express algorithms in terms of operations on immutable data structures using pure functions.
2. Written in this day, useful program transformations (differentiation, compilation, vectorisation, etc.) may be automatically applied by the framework without further intervention.

Most of these complications are not exposed to end users, but being aware of this is important if attempting to build on top of this library.


### Adding custom architectures

There are multiple routes to add new architectures for approximation of various tensor fields. The simplest one is just to provide a Jax function, but the recommended route, keeping in line with the logic in the models module, is to add:

1. A [Flax module](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html) describing the sequence of operations defined by your architecture.
```py
import jax.numpy as jnp
from flax import linen as nn

class MyAnsatz(nn.Module):
    # toy example
    def setup(self):
        self.layer = nn.Einsum(...)   # some logic

    @nn.compact
    def __call__(self, local_coords):
        p = local_coords
        p_bar = jnp.conjugate(p)
        p_norm_sq = jnp.sum(p * p_bar)
        return jnp.outer(p, p_bar) / p_norm_sq + self.layer(p)
```
2. A pure function which accepts a pytree of parameters for the model and executes the computation by invoking the `.apply` method of the module you defined above.
```py
def tensor_ansatz(p, params, *args):
    p = ...  # some logic
    model = MyAnsatz(*args)  # model constructor
    return model.apply({'params': params}, p)
```

### Downstream computations
You have run some optimisation procedure, obtaining a parameterised function which approximates some tensor field in local coordinates. For concreteness, let us say this is the metric tensor. As it is likely that any downstream computation will involve some differential operator, it is recommended to apply a partial closure, binding all arguments except for the coordinate dependency. 

It is recommended to use `Jax`'s [pytree-compatible partial evaluation](https://jax.readthedocs.io/en/latest/_autosummary/jax.tree_util.Partial.html) instead of the conventional `functools.partial` call, such that the function may be passed as an argument to transformed `Jax` functions.
```py
import jax
import jax.numpy as jnp

def approx_metric_fn(p, params, *args):
    g = ... # some logic
    return g

@jax.jit
def christoffel_symbols(p, metric_fn):
    g_inv = jnp.linalg.inv(metric_fn(p))
    jac_g_holo = del_z(p, metric_fn)
    return jnp.einsum('...kl, ...jki->...lij', g_inv, jac_g_holo)

metric_fn = jax.tree_util.Partial(approx_metric_fn, params, *args)
Gamma = christoffel_symbols(p, metric_fn)
```

### Functions accessing global state
Because useful program transformations assume that the functions they act on are pure, functions which read or write to global state can [result in undefined behaviour](https://jax.readthedocs.io/en/latest/stateful-computations.html). The simplest way to resolve this is to manually carry around arguments to functions. This is clunky in general and may be alleviated through a partial closure for static arguments, using [`functools.partial`](https://docs.python.org/3/library/functools.html#functools.partial) or [`tree_util.partial`](https://docs.python.org/3/library/functools.html#functools.partial) for compatibility with program transformations. Another alternative is to use filtered transformations, as in [Equinox](https://docs.kidger.site/equinox/). 


### The compiler throws an arcane error
Most of the time, this is due to:

* Program logic violating the constraints placed by the XLA compiler, and the resolution can be found in [this compendium](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html).
* Memory issues when computing curvature quantities which involve higher-order derivatives of some neural network architecture with respect to the input points. In this case try reducing the `vmap` batch size or decrease the complexity of the architecture.

However, there can be a few truly head-scratching errors. In that case, please raise an issue or feel free to contact us.

### Miscellanea
Dev notes that don't fit anywhere else.

* The documentation uses the [jaxtyping](https://docs.kidger.site/jaxtyping/api/array/) conventions for array annotations.
* A good chunk of code is not exposed to the public API as it is mostly for internal purposes or privileged downstream packages. Please get in touch if the comments are insufficient and you want the docs to be expanded.