import tensorflow as tf
from gpflow.kernels import SquaredExponential, Kernel, Matern52
from tensorflow import Tensor
from gpflow.base import TensorLike
from .dispatch import magic_Kuu, magic_Kuf
from ..inducing_variables.inducing_variables import VariableInducingPoints

@magic_Kuu.register(VariableInducingPoints, Kernel)
def magic_K_fallback(inducing_variable: VariableInducingPoints, kernel: Kernel):
    prior_cov = kernel(inducing_variable.Z)
    condition = tf.linalg.inv(kernel(inducing_variable.ny))
    left_term = kernel(inducing_variable.Z, inducing_variable.ny)
    return prior_cov - left_term @ condition @ tf.transpose(left_term)

@magic_Kuf.register(VariableInducingPoints, Kernel, TensorLike)
def magic_K_fallback(inducing_variable: VariableInducingPoints, kernel: Kernel, Xnew):
    prior_cov = kernel(inducing_variable.Z, Xnew)
    condition = tf.linalg.inv(kernel(inducing_variable.ny))
    left_term = kernel(inducing_variable.Z, inducing_variable.ny)
    right_term = kernel(inducing_variable.ny, Xnew)
    return prior_cov - left_term @ condition @ right_term