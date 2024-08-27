import jax
import jax.numpy as jnp

from flax import linen as nn
from typing import (
  Any,
  List,
  Optional,
)

import jax
import jax.numpy as jnp

from flax.linen import initializers
from flax.linen.dtypes import promote_dtype
from flax.linen import module
from flax.linen.module import Module, compact
from flax.typing import (
  Array,
  PRNGKey as PRNGKey,
  Dtype,
  Shape as Shape,
  Initializer,
  PrecisionLike,
)

default_kernel_init = initializers.lecun_normal()


class EinsumComplex(nn.Einsum):
  """An einsum transformation with learnable complex kernel constructed from
     real, imaginary parts, and bias.

  Example usage::

    >>> import flax.linen as nn
    >>> import jax, jax.numpy as jnp

    >>> layer = nn.Einsum((5, 6, 7), 'abc,cde->abde')
    >>> variables = layer.init(jax.random.key(0), jnp.ones((3, 4, 5)))
    >>> jax.tree_util.tree_map(jnp.shape, variables)
    {'params': {'bias': (6, 7), 'kernel': (5, 6, 7)}}

  Attributes:
    shape: the shape of the kernel.
    einsum_str: a string to denote the einsum equation. The equation must
      have exactly two operands, the lhs being the input passed in, and
      the rhs being the learnable kernel. Exactly one of ``einsum_str``
      in the constructor argument and call argument must be not None,
      while the other must be None.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see ``jax.lax.Precision``
      for details.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
  """

  shape: Shape
  einsum_str: Optional[str] = None
  use_bias: bool = True
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None
  kernel_init: Initializer = default_kernel_init
  bias_init: Initializer = initializers.zeros_init()

  @compact
  def __call__(self, inputs: Array, einsum_str: Optional[str] = None) -> Array:
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.
      einsum_str: a string to denote the einsum equation. The equation must
        have exactly two operands, the lhs being the input passed in, and
        the rhs being the learnable kernel. Exactly one of ``einsum_str``
        in the constructor argument and call argument must be not None,
        while the other must be None.

    Returns:
      The transformed input.
    """
    einsum_str = module.merge_param('einsum_str', self.einsum_str, einsum_str)

    einsum_str = einsum_str.replace(' ', '')
    if '->' not in einsum_str:
      raise ValueError(
        '`einsum_str` equation must be explicit and include "->".'
      )
    if einsum_str.count(',') != 1:
      raise ValueError(
        '`einsum_str` equation must have exactly two operands and '
        'therefore, exactly one comma character, instead of '
        f'{einsum_str.count(",")}'
      )

    re_kernel = self.param(
      're_kernel',
      self.kernel_init,
      self.shape,
      self.param_dtype,
    )

    im_kernel = self.param(
      'im_kernel',
      self.kernel_init,
      self.shape,
      self.param_dtype,
    )

    kernel = jax.lax.complex(re_kernel, im_kernel)

    if self.use_bias:
      bias_shape, broadcasted_bias_shape = self._get_bias_shape(
        einsum_str, inputs, kernel
      )
      bias = self.param('bias', self.bias_init, bias_shape) # , self.param_dtype)
    else:
      bias = None

    inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)

    y = jnp.einsum(einsum_str, inputs, kernel, precision=self.precision)

    if bias is not None:
      y += jnp.reshape(bias, broadcasted_bias_shape)
    return y