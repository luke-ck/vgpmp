from typing import Union

import tensorflow as tf
from gpflow.base import TensorLike
from gpflow.kernels.multioutput import SeparateIndependent, IndependentLatent
from gpflow import default_jitter
from tensorflow.python.trackable.data_structures import ListWrapper

# from keras.utils.tf_utils import ListWrapper

from ..dispatch import K_conditioned
from typing import Sequence

from ...kernels.kernels import VanillaConditioningSeparateIndependent


@K_conditioned.register(TensorLike, TensorLike, (SeparateIndependent, IndependentLatent))
def k_cond_fallback_shared(Z: TensorLike,
                           X: TensorLike,
                           kernel: Union[SeparateIndependent, IndependentLatent]) -> tf.Tensor:
    K = tf.stack(
        [K_conditioned(Z[..., idx], X[..., idx], k) for idx, k in enumerate(kernel.kernels)], axis=0
    )  # [L, Z, X]

    return K


@K_conditioned.register(ListWrapper, TensorLike, (SeparateIndependent, IndependentLatent))
def kuf_cond_fallback_separate(Z: ListWrapper,
                               X: TensorLike,
                               kernel: Union[SeparateIndependent, IndependentLatent]) -> tf.Tensor:
    K = tf.stack(
        [K_conditioned(Z[idx], X, k) for idx, k in enumerate(kernel.kernels)], axis=0
    )  # [L, Z, X]

    return K


@K_conditioned.register(ListWrapper, (SeparateIndependent, IndependentLatent, VanillaConditioningSeparateIndependent))
def kuu_cond_fallback_shared_list(Z: ListWrapper,
                                  kernel: Union[
                                      SeparateIndependent, IndependentLatent, VanillaConditioningSeparateIndependent]) -> tf.Tensor:
    K = tf.stack(
        [K_conditioned(Z[idx], Z[idx], k) for idx, k in enumerate(kernel.kernels)], axis=0
    )  # [L, Z, Z]

    # jittermat = tf.eye(K.shape[-1], dtype=K.dtype)[None, :, :] * default_jitter()
    return K  # + jittermat
