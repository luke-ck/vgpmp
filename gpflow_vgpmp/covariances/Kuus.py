import tensorflow as tf
from gpflow.covariances.dispatch import Kuu
from gpflow.kernels import SquaredExponential

from ..inducing_variables.inducing_variables import VariableInducingPoints


@Kuu.register(VariableInducingPoints, SquaredExponential)
def Kuu_kernel_inducingpoints(inducing_variable: VariableInducingPoints, kernel: SquaredExponential, *, jitter=0.0):
    Kzz = kernel(inducing_variable.Zy)  # this computes k([u, y].T)
    Kzz += jitter * tf.eye(inducing_variable.num_inducing + 2, dtype=Kzz.dtype)
    return Kzz
