from typing import Union

import tensorflow as tf
from gpflow.base import TensorLike
from gpflow.kernels.multioutput import SeparateIndependent, IndependentLatent

from ..dispatch import K_conditioned


@K_conditioned.register(TensorLike, TensorLike, (SeparateIndependent, IndependentLatent))
def k_cond_fallback_shared(Z: TensorLike,
                           X: TensorLike,
                           kernel: Union[SeparateIndependent, IndependentLatent]) -> tf.Tensor:
    K = tf.stack(
        [K_conditioned(Z[..., idx], X[..., idx], k) for idx, k in enumerate(kernel.kernels)], axis=0
    )  # [L, Z, X]

    return K
