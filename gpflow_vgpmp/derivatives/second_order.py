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


@K_grad_grad.register(Tensor, Matern52)
def k_grad_grad_se_fallback(
        inducing_location_ny: Tensor,
        kernel: Matern52,
):
    iterator = tf.range(inducing_location_ny.shape[0])
    block = tf.map_fn(lambda i: second_order_derivative_se(inducing_location_ny[i], inducing_location_ny, kernel),
                      iterator, fn_output_signature=tf.float64, parallel_iterations=8)

    return block
