# Getting started

`cymyc` is a library for numerical differential geometry on Calabi-Yau manifolds written in JAX, enabling performant:

* Approximations of useful tensor fields;
* Computations of curvature-related quantities;
* Investigations of the complex structure moduli space;

in addition to many other features. 

If you're completely new to JAX or Python, then start with this [EXAMPLE.](basics.md)

## Citation
--8<-- ".citation.md"

## Related libraries

**Numerical metrics on Calabi-Yaus**

* [cymetric]() - Python library for studying Calabi-Yau metrics.
* [MLGeometry](https://github.com/yidiq7/MLGeometry) - Machine learning Calabi-Yau metrics.

**JAX ecosystem**

* [equinox](https://github.com/patrick-kidger/equinox) - JAX enhancement suite.
* [flax](https://github.com/google/flax) - Neural network library.


## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
