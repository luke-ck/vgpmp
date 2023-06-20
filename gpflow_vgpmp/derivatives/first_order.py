import tensorflow as tf
from gpflow.kernels.stationaries import SquaredExponential, Kernel, Matern52
from tensorflow import Tensor

from .dispatch import K_grad


SQRT_5 = 2.2360679774997898
FIVE_THIRDS = 1.6666666666666667

@K_grad.register(Tensor, Tensor, SquaredExponential)
def k_grad_se_fallback(inducing_variable_ny, inducing_variable_Zy, kernel):
    norm = 1 / (kernel.lengthscales ** 2)
    partial_derivative = norm * tf.subtract(tf.expand_dims(inducing_variable_ny, 1), inducing_variable_Zy)
    return partial_derivative * kernel(inducing_variable_ny[..., None], inducing_variable_Zy[..., None])


@K_grad.register(Tensor, Tensor, Matern52)
def k_grad_matern52_fallback(inducing_variable_ny, inducing_variable_Zy, kernel):
    # TODO: implement this
    r2 = kernel.scaled_squared_euclid_dist(inducing_variable_ny[..., None], inducing_variable_Zy[..., None])
    dkernel_dr_over_r = -FIVE_THIRDS * (1 + SQRT_5 * tf.sqrt(r2)) * tf.exp(-SQRT_5 * tf.sqrt(r2))
    return dkernel_dr_over_r * tf.subtract(tf.expand_dims(inducing_variable_ny, 1), inducing_variable_Zy) / kernel.variance
    # return norm * partial_derivative

@K_grad.register(Tensor, Tensor, Kernel)
def k_grad_matern72_fallback(inducing_variable_ny, inducing_variable_Zy, kernel):
    # TODO: implement this
    raise NotImplementedError

@tf.function
def kernel_derivative(ny, Zy, kernel):
    ny_block = K_grad(ny, ny, kernel)
    nyZy_block = K_grad(ny, Zy, kernel)
    Zy_block = K_grad(Zy, Zy, kernel)
    upper_block = tf.concat([ny_block, nyZy_block], axis=1)
    lower_block = tf.concat([-tf.transpose(nyZy_block), Zy_block], axis=1)
    return tf.concat([upper_block, lower_block], axis=0)


