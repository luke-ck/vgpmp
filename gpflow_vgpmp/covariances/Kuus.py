import tensorflow as tf
from gpflow.covariances.dispatch import Kuu
from gpflow.kernels import SquaredExponential, Kernel

from ..inducing_variables.inducing_variables import InducingPoints


@Kuu.register(InducingPoints, Kernel)
def Kuu_kernel_inducingpoints(inducing_variable: InducingPoints, kernel: SquaredExponential, *, jitter=0.0):
    Kzz = kernel(inducing_variable.Z)  # this computes k([u, y].T)
    Kzz += jitter * tf.eye(inducing_variable.num_inducing + 2, dtype=Kzz.dtype)
    return Kzz

