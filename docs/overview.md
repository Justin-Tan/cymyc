---
icon: material/pyramid
---

# Overview
This library is a tool for numerical differential geometry, with a focus on Kähler geometry and calculations related to string compactifications. 

## The `Jax` computational paradigm
`Jax` is a high-performance Python library for program transformations - chief among these being automatic differentiation. This is the transformation of a program into another, 
$$
\texttt{program} \rightarrow \partial(\texttt{program})~,
$$ 
which evaluates the partial derivative with respect to any of $\texttt{program}$'s original inputs. From a computational geometry perspective, this is a boon for computation of curvature-related quantities and differential operators on manifolds. For example, given a program which outputs the metric in local coordinates, one schematically arrives at the various horsemen of curvature via derivatives w.r.t. local coordinates,

\[ 
    \varphi \stackrel{\partial}{\rightarrow} \left(g_{\mu \nu} \sim \partial \partial \varphi \right) \stackrel{\partial}{\rightarrow} \left(\Gamma^{\kappa}_{\mu \nu} \sim g \cdot \partial g\right) \stackrel{\partial}{\rightarrow} \left(R^{\kappa}_{\, \, \lambda \mu \nu} \sim \partial \Gamma + \Gamma \cdot \Gamma\right) \rightarrow \cdots 
\]

## Philosophy

The pullback metric $\iota^* g$ may be computed as:

\[
    \iota^* g = J^T g J\,.
\]

### abstract nonsense

```py title="curvature"
import jax
import jax.numpy as jnp

import numpy as np

def metric_fn(p):
    return 

def christoffel_symbols(p):
    return del_z(p, metric_fn)
```

!!! tip
    This library is device-agnostic. That being said, autodiff routines will usually be significantly faster if the user has access to a GPU. If this applies to you, follow the Jax [GPU installation instructions](https://github.com/google/jax?tab=readme-ov-file#installation) to enable GPU support.


!!! note
    This is a note.

!!! tip
    This is a tip.

!!! info
    and this is an info block.
