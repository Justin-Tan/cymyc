import typing as tp
import jax
from typing import TypeVar
from dataclasses import dataclass

from jaxtyping import Array, Float, Complex, ArrayLike

T = TypeVar('T')

# Type aliases
M = tp.TypeVar('M', bound=jax.Array)  # Points on a manifold
TpM = tp.TypeVar('TpM', bound=jax.Array)  # Vectors in tangent space
MetricFn = tp.Callable[[Array], Array]


@dataclass
class TangentSpace(tp.Generic[M, TpM]):
    """Representation of the tangent space on the manifold
    Parameters:
        point: position p ∈ M
        vector: corresponding vector, v ∈ T_p M
    """
    point: M
    vector: TpM

