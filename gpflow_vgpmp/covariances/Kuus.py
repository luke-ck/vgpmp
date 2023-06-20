import tensorflow as tf
from gpflow.covariances.dispatch import Kuu
from gpflow.kernels import SquaredExponential, Kernel

from ..inducing_variables.inducing_variables import VariableInducingPoints, InducingPointsInterface


@Kuu.register(InducingPointsInterface, Kernel)
def Kuu_kernel_inducingpoints(inducing_variable: VariableInducingPoints, kernel: SquaredExponential, *, jitter=0.0):
    padding = 2
    Kzz = kernel(inducing_variable.Zy)  # this computes k([u, y].T)
    Kzz += jitter * tf.eye(inducing_variable.num_inducing + padding, dtype=Kzz.dtype)
    return Kzz
