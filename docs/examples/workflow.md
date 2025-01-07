# Example workflow

If you have a particular Calabi-Yau in mind you want to investigate, this example walks you through how to do this using the main scripts in the library. The arguments for each script should be amply documented, accessible using the `-h` flag. To begin, navigate to the root directory of the repository.

## Manifold definition / point sampling

Here we take as example the mirror to the manifold $\mathbb{P}^5[3,3]$. This is defined as the intersection of the zero loci of the following polynomials in $\mathbb{P}^5$,

\begin{align*}
    P_1 &= Z_0^3 + Z_1^3 + Z_2^3 - 3 \psi Z_3 Z_4 Z_5~, \\
    P_2 &= Z_3^3 + Z_4^3 + Z_5^3 - 3 \psi Z_0 Z_1 Z_2~.
\end{align*}


 This manifold has a single complex structure modulus - see [this article](https://arxiv.org/abs/1903.00596) for more details. We choose the point $\psi = 1/2$ in complex structure moduli space. The following script samples 320000 points from the above zero locus, together with 64000 validation points, saving it to the directory `data/X33`. 

 !!! info
    Point sampling occurs with respect to a given choice of the complex structure moduli, currently one may consider the complex structure of the approximate metric fixed by this stage. 

```python
python3 -m cymyc.utils.pointgen_cicy -o data/X33 -n_p 320000 -val 0.2 -psi 0.5
```

To register your favorite manifold with this library, you will need to know the zero locus it traces out. add the definition to `examples/poly_spec.py`, following the conventions within, and reference it as appropriate downstream.

## Metric optimisation
To obtain an approximation to the Ricci-flat metric on $\mathbb{P}^5[3,3]$, run the following script:

```python
python3 -m cymyc.approx.train -name X33_metric -ds data/X33
```
This will output the parameters of the neural network approximating the Ricci-flat metric to the directory `experiments/X33_metric/`. One may now compute geometric quantities using this approximate metric using the routines in `src/curvature`, or compute exponential maps / geodesics etc. using the [`diffrax`](https://docs.kidger.site/diffrax/) library. 


## Harmonic form approximation
To approximate harmonic differential forms and thereby Yukawa couplings on $\mathbb{P}^5[3,3]$, run the following script, supplying the path to the model checkpoint of the approximate metric saved at the end of the optimisation process, found under `experiments/X33_metric`. 

```python
python3 -m cymyc.approx.eta_train -name X33_harmonic -ds data/X33 -ckpt /path/to/metric/checkpoint
```

More information about the library may be found in the API docs - feel free to get in touch with any questions.

## Complex structure moduli
One may obtain the Weil--Petersson metric over complex structure moduli space for Calabi-Yau manifolds with any number of complex structure moduli via the routines in `src/moduli/moduli_scan`. This computation is based on deformation theory and does not involve period integrals or any approximation. It is exact up to Monte Carlo integration error, and serves as a sanity check for the approximations to the moduli metric obtained using harmonic forms.

See `examples/tian_yau` for scripts which evaluate the moduli space metric over moduli space with dimension $h^{2,1} = 9$.
