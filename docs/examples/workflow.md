# Example workflow

If you have a particular Calabi-Yau in mind you want to investigate, this example walks you through how to do this using the main scripts in the library. The arguments for each script should be amply documented, accessible using the `-h` flag. To begin, navigate to the root directory of the repository.

## Manifold definition / point sampling

Here we take as example the mirror to the manifold $\mathbb{P}^5[2,4]$. This is defined as the intersection of the zero loci of the following polynomials in $\mathbb{P}^5$,

\begin{align*}
    p_1 &= Z_1^2 + Z_2^3 +Z_3^3 - 3 \psi Z_4Z_5Z_6~, \\
    p_2 &= Z_4^3 + Z_5^3 +Z_6^3 - 3 \psi Z_1Z_2Z_3~.
\end{align*}

 This manifold has a single complex structure modulus - see [this article](https://arxiv.org/abs/1903.00596) for more details. We choose the point $\psi = 1/2$ in complex structure moduli space. The following script samples 320000 points from the above zero locus, together with 64000 validation points, saving it to the directory `data/X24`. 

```python
python3 -m cymyc.utils.pointgen_cicy -o data/X24 -n_p 320000 -val 0.2 -psi 0.5
```

To register your favorite manifold with this library, you will need to know the zero locus it traces out. add the definition to `examples/poly_spec.py`, following the conventions within, and reference it as appropriate downstream.

## Metric optimisation
To obtain an approximation to the Ricci-flat metric on $\mathbb{P}^5[2,4]$, run the following script:

```python
python3 -m cymyc.approx.train -name X24_metric -ds data/X24
```
This will output the parameters of the neural network approximating the Ricci-flat metric to the directory `experiments/X24_psi/`. One may now compute geometric quantities using this approximate metric using the routines in `src/curvature`, or compute exponential maps / geodesics etc. using the [`diffrax`](https://docs.kidger.site/diffrax/) library. 

## Harmonic form approximation
To approximate harmonic differential forms on $\mathbb{P}^5[2,4]$, run the following script, supplying the path to the model checkpoint of the approximate metric saved at the end of the optimisation process.

```python
python3 -m cymyc.approx.eta_train -name X24_harmonic -ds data/X24 -ckpt /path/to/metric/checkpoint
```

More information about the library may be found in the API docs - feel free to get in touch with any questions.

## Complex structure moduli
One may obtain the Weil--Petersson metric over complex structure moduli space for Calabi-Yau manifolds with any number of complex structure moduli via the routines in `src/moduli/moduli_scan`. This computation is based on deformation theory and does not involve period integrals or any approximation. It is exact up to Monte Carlo integration error, and serves as a sanity check for the approximations to the moduli metric obtained using harmonic forms.

See `examples/tian_yau` for scripts which evaluate the moduli space metric over moduli space with dimension $h^{2,1} = 9$.
