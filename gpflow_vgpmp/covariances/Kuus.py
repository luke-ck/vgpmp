import tensorflow as tf
from gpflow.covariances.dispatch import Kuu
from gpflow.kernels import Kernel

from ..inducing_variables.inducing_variables import InducingPointsInterface


@Kuu.register(InducingPointsInterface, Kernel)
def Kuu_kernel_inducingpoints(inducing_variable: InducingPointsInterface, kernel: Kernel, *, jitter=0.0):
    padding = inducing_variable.ny.shape[0]
    Kzz = kernel(inducing_variable.Zy)  # this computes k([u, y].T)
    Kzz += jitter * tf.eye(inducing_variable.num_inducing + padding, dtype=Kzz.dtype)
    return Kzz
