import tensorflow as tf
from gpflow.kernels.stationaries import Kernel
from gpflow.base import TensorLike
from .dispatch import K_conditioned


@K_conditioned.register(TensorLike, TensorLike, Kernel)
def k_cond_se_fallback(Z, X, kernel):
    K2 = kernel(Z[..., None], X[..., None])
    return K2




