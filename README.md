# CYMYC

`cymyc` is a library for numerical differential geometry on Calabi-Yau manifolds written in JAX, enabling performant:

* Approximations of useful tensor fields;
* Computations of curvature-related quantities;
* Investigations of the complex structure moduli space;

in addition to many other features. 

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
Requires Python >=3.10 - all needed dependencies will be automatically installed.

## Documentation / examples

Check out the docs at [justin-tan.github.io/cymyc/](https://justin-tan.github.io/cymyc/). If you are new to Jax and want to get your hands dirty, then start with [this example](https://justin-tan.github.io/cymyc/examples/curvature/). Also see the [example workflow](https://justin-tan.github.io/cymyc/examples/workflow/) for canonical usage of the scripts in this library.


### Contributing / Development
This library is under active development, and the current state is but the leading order approximation. Please open an issue / pull request if you encounter unexpected behaviour. Additionally, feel free to get in touch anytime to discuss the project and help us guide development.

## Citation
If you found this library to be useful in academic work, then please cite: ([arxiv](https://arxiv.org/abs/2410.19728)).

```bibtex
@article{cymyc,
	author = {Butbaia, Giorgi, Tan, Justin, ...},
	title = {\textsf{cymyc}: {\it A \textsf{{JAX}} package for {C}alabi--{Y}au 
	{M}etrics {Y}ukawas and {C}urvature}},
    eprint = "2410.19728",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    month = "10",
    year = "2024"
}
```

## Related work

This codebase was used to generate the 'experimental' results for the following publications:

1. Curvature on Calabi-Yau manifolds - [arxiv:2211.90801](https://arxiv.org/abs/2211.09801).
2. Physical Yukawa couplings in heterotic string compactifications - [arxiv:2401.15078](https://arxiv.org/abs/2401.15078).
3. Precision string phenomenology - [arxiv:2407.13836](https://arxiv.org/abs/2407.13836).


## Related libraries / Acknowledgements

We would like to acknowledge the authors of the [cymetric](https://github.com/pythoncymetric/cymetric) library ([Larfors et. al. (2022)](https://iopscience.iop.org/article/10.1088/2632-2153/ac8e4e/meta)).

**Numerical metrics on Calabi-Yaus**

* [cymetric](https://github.com/pythoncymetric/cymetric) - Python library for studying Calabi-Yau metrics.
* [cyjax](https://github.com/ml4physics/cyjax) - Donaldson's algorithm for Calabi-Yau metrics in Jax.
* [MLGeometry](https://github.com/yidiq7/MLGeometry) - Machine learning Calabi-Yau metrics.

**JAX ecosystem**

* [equinox](https://github.com/patrick-kidger/equinox) - JAX enhancement suite.
* [flax](https://github.com/google/flax) - Neural network library.


