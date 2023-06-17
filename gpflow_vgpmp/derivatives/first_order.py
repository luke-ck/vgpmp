import tensorflow as tf
from gpflow.kernels.stationaries import SquaredExponential, Kernel, Matern52
from tensorflow import Tensor

from .dispatch import K_grad


@K_grad.register(Tensor, Tensor, SquaredExponential)
def k_grad_se_fallback(inducing_variable_ny, inducing_variable_Zy, kernel):
    norm = 1 / (kernel.lengthscales ** 2)
    partial_derivative = norm * tf.subtract(tf.expand_dims(inducing_variable_ny, 1), inducing_variable_Zy)
    return partial_derivative


@K_grad.register(Tensor, Tensor, Matern52)
def k_grad_matern52_fallback(inducing_variable_ny, inducing_variable_Zy, kernel):
    funcs2 = tf.range(inducing_variable_Zy.shape[0])
    norm = 1 / (kernel.lengthscales ** 2)
    partial_derivative = tf.map_fn(
        lambda j: (inducing_variable_ny - inducing_variable_Zy[j]),
        funcs2, fn_output_signature=tf.float64, parallel_iterations=8)
    return norm * partial_derivative

@tf.function
def kernel_derivative(ny, Zy, kernel):
    ny_block = K_grad(ny, ny, kernel)
    nyZy_block = K_grad(ny, Zy, kernel)
    Zy_block = K_grad(Zy, Zy, kernel)
    upper_block = tf.concat([ny_block, nyZy_block], axis=1)
    lower_block = tf.concat([-tf.transpose(nyZy_block), Zy_block], axis=1)
    return tf.concat([upper_block, lower_block], axis=0)


