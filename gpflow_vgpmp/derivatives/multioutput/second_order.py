from typing import Union

import tensorflow as tf
from gpflow.base import TensorLike
from gpflow.kernels.multioutput import SeparateIndependent, IndependentLatent

from ..dispatch import K_grad_grad


@K_grad_grad.register(TensorLike, (SeparateIndependent, IndependentLatent))
def k_grad_grad_fallback_shared(Z: TensorLike,
                                kernel: Union[SeparateIndependent, IndependentLatent]) -> tf.Tensor:
    KyZy = tf.stack(
        [K_grad_grad(Z[..., idx], k) for idx, k in enumerate(kernel.kernels)], axis=0
    )  # [L, 2, 2]
    K = tf.stack([1 / (k.lengthscales ** 4) * k(Z) for k in kernel.kernels], axis=0)
    return tf.multiply(KyZy, K)
