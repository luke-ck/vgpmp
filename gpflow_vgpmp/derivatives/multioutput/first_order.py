from typing import Union

import tensorflow as tf
from gpflow.base import TensorLike
from gpflow.kernels.multioutput import SeparateIndependent, IndependentLatent

from ..dispatch import K_grad


@K_grad.register(TensorLike, TensorLike, (SeparateIndependent, IndependentLatent))
def k_grad_fallback_shared(Z: TensorLike,
                           X: TensorLike,
                           kernel: Union[SeparateIndependent, IndependentLatent]) -> tf.Tensor:
    KyZy = tf.stack(
        [K_grad(Z[..., idx], X[..., idx], k) for idx, k in enumerate(kernel.kernels)], axis=0
    )  # [L, Z, X]
    K = tf.stack([k(Z, X) for k in kernel.kernels], axis=0)
    return tf.multiply(KyZy, K)
