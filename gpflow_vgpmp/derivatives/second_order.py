import tensorflow as tf
from gpflow.kernels.stationaries import SquaredExponential, Kernel, Matern52
from tensorflow import Tensor

from .dispatch import K_grad_grad


@tf.function
def second_order_derivative_se(inducing_variable_ny_scalar, inducing_variable_ny, kernel):
    funcs2 = tf.range(inducing_variable_ny.shape[0])
    norm = kernel.lengthscales ** 2
    second_derivative = tf.map_fn(
        lambda j: (norm - (inducing_variable_ny_scalar - inducing_variable_ny[j]) ** 2)
        , funcs2, fn_output_signature=tf.float64, parallel_iterations=8)
    return second_derivative


@K_grad_grad.register(Tensor, SquaredExponential)
def k_grad_grad_se_fallback(
        inducing_location_ny: Tensor,
        kernel: SquaredExponential,
):
    norm = kernel.lengthscales ** 2
    inducing_diff = tf.expand_dims(inducing_location_ny, 1) - tf.expand_dims(inducing_location_ny, 0)
    second_derivative = norm - inducing_diff ** 2
    return second_derivative / kernel.lengthscales ** 4
