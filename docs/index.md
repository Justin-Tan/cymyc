---
icon: material/blur-radial
---

# Getting started

`cymyc` is a library for numerical differential geometry on Calabi-Yau manifolds written in JAX, enabling performant:

* Approximations of useful tensor fields;
* Computations of curvature-related quantities;
* Investigations of the complex structure moduli space;

in addition to many other features. 

If you are new to Jax and want to get your hands dirty, then start with [this example](examples/curvature.ipynb).

## Installation

First, clone the project:

```shell
git clone git@github.com:Justin-Tan/cymyc.git
cd cymyc
```

Next, with a working Python installation, create a new virtual environment and run an editable install, which permits local development.

```shell
pip install --upgrade pip
python -m venv /path/to/venv
source /path/to/venv/bin/activate

python -m pip install -e .
```
Requires Python >=3.10 - all needed dependencies will be automatically installed. See this example on how to use [the main scripts](examples/workflow.md).

!!! tip
    This library is device-agnostic. That being said, autodiff routines will usually be significantly faster if the user has access to a GPU. If this applies to you, follow the Jax [GPU installation instructions](https://github.com/google/jax?tab=readme-ov-file#installation) to enable GPU support.

### Contributing / Development
This library is under active development, and the current state is but the leading order approximation. Please open an issue / pull request if you encounter unexpected behaviour. Additionally, feel free to get in touch anytime to discuss the project or help us guide development.

## Citation
--8<-- ".citation.md"

## Related work

This codebase was used to generate the 'experimental' results for the following publications:

1. Curvature on Calabi-Yau manifolds - [arxiv:2211.90801](https://arxiv.org/abs/2211.09801).
2. Physical Yukawa couplings in heterotic string compactifications - [arxiv:2401.15078](https://arxiv.org/abs/2401.15078).
3. Precision string phenomenology - [arxiv:2407.13836](https://arxiv.org/abs/2407.13836).

## Related libraries / Acknowledgements

We would like to acknowledge the authors of the [cymetric](https://github.com/pythoncymetric/cymetric) library ([Larfors et. al. (2022)](https://iopscience.iop.org/article/10.1088/2632-2153/ac8e4e/meta)), whose excellent work this library builds upon.

**Numerical metrics on Calabi-Yaus**

* [cymetric](https://github.com/pythoncymetric/cymetric) - Python library for studying Calabi-Yau metrics.
* [cyjax](https://github.com/ml4physics/cyjax) - Donaldson's algorithm for Calabi-Yau metrics in Jax.
* [MLGeometry](https://github.com/yidiq7/MLGeometry) - Machine learning Calabi-Yau metrics

**JAX ecosystem**

* [equinox](https://github.com/patrick-kidger/equinox) - JAX enhancement suite.
* [flax](https://github.com/google/flax) - Neural network library.


