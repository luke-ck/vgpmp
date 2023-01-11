import tensorflow as tf
from gpflow.kernels.stationaries import SquaredExponential, Kernel, Matern52
from tensorflow import Tensor

from .dispatch import K_grad


@tf.function
def first_order_derivative_se(inducing_variable_ny, inducing_variable_Zy, kernel):
    funcs2 = tf.range(inducing_variable_Zy.shape[0])
    norm = 1 / (kernel.lengthscales ** 2)
    partial_derivative = tf.map_fn(
        lambda j: (inducing_variable_ny - inducing_variable_Zy[j]),
        funcs2, fn_output_signature=tf.float64, parallel_iterations=8)
    return norm * partial_derivative


@K_grad.register(Tensor, Tensor, Matern52)
def k_grad_se_fallback(
        inducing_location_ny: Tensor,
        inducing_location_Zy: Tensor,
        kernel: Matern52,
):
    iterator = tf.range(inducing_location_ny.shape[0])
    block = tf.map_fn(lambda i: first_order_derivative_se(inducing_location_ny[i], inducing_location_Zy, kernel),
                      iterator, fn_output_signature=tf.float64, parallel_iterations=8)

    return block
