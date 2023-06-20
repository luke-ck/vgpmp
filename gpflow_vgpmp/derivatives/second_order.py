import tensorflow as tf
from gpflow.kernels.stationaries import SquaredExponential, Kernel, Matern52
from tensorflow import Tensor

from .dispatch import K_grad_grad, K_grad


SQRT_5 = 2.2360679774997898
FIVE_THIRDS = 1.6666666666666667
@tf.function
def second_order_derivative_se(inducing_variable_ny_scalar, inducing_variable_ny, kernel):
    funcs2 = tf.range(inducing_variable_ny.shape[0])
    norm = kernel.lengthscales ** 2
    second_derivative = tf.map_fn(
        lambda j: (norm - (inducing_variable_ny_scalar - inducing_variable_ny[j]) ** 2)
        , funcs2, fn_output_signature=tf.float64, parallel_iterations=8)
    return second_derivative


@K_grad_grad.register(Tensor, Matern52)
def k_grad_grad_matern52_fallback(
        inducing_location_ny: Tensor,
        kernel: Matern52,
):
    r2 = kernel.scaled_squared_euclid_dist(inducing_location_ny[..., None], inducing_location_ny[..., None])
    r = tf.sqrt(r2)
    diff = tf.expand_dims(inducing_location_ny, 1) - tf.expand_dims(inducing_location_ny, 0)
    dr_dXn = diff / (r[:, tf.newaxis] * kernel.variance)
    dr_dYm = -diff / (r[:, tf.newaxis] * kernel.variance)
    d2r_dXndYm_times_r3 = diff[:, :, tf.newaxis] * diff[:, tf.newaxis, :] / (kernel.variance * kernel.variance)
    d2r_dXndYm_times_r3 -= tf.eye(d2r_dXndYm_times_r3.shape[1]) * (r ** 2 / kernel.variance)
    s5r = SQRT_5 * r
    exp_minus_s5r = tf.exp(-s5r)
    dkernel_dr_over_r = -FIVE_THIRDS * (1 + s5r) * exp_minus_s5r
    d2kernel_dr2 = FIVE_THIRDS * (5 * r2 - s5r - 1) * exp_minus_s5r
    term1 = dkernel_dr_over_r[:, tf.newaxis, tf.newaxis] * d2r_dXndYm_times_r3 / r2[:, tf.newaxis, tf.newaxis]
    term2 = d2kernel_dr2[:, tf.newaxis, tf.newaxis] * dr_dXn[:, :, tf.newaxis] * dr_dYm[:, tf.newaxis, :]
    return term1 + term2

@K_grad_grad.register(Tensor, SquaredExponential)
def k_grad_grad_se_fallback(
        inducing_location_ny: Tensor,
        kernel: SquaredExponential,
):
    norm = kernel.lengthscales ** 2
    inducing_diff = tf.expand_dims(inducing_location_ny, 1) - tf.expand_dims(inducing_location_ny, 0)
    second_derivative = norm - inducing_diff ** 2
    return second_derivative / kernel.lengthscales ** 4 * kernel(inducing_location_ny[..., None], inducing_location_ny[..., None])


@tf.function
def k_grad_grad(ny, Zy, kernel):
    block_original = kernel(Zy)[..., None]
    ny_block = K_grad_grad(ny, kernel) * block_original[:ny.shape[0], :ny.shape[0]]
    nyZy_block = K_grad(ny, Zy[ny.shape[0]:], kernel) * block_original[:ny.shape[0], ny.shape[0]:]
    upper_block = tf.concat([ny_block, nyZy_block], axis=1)
    lower_block = tf.concat([-tf.transpose(nyZy_block, perm=(1, 0, 2)), block_original[ny.shape[0]:, ny.shape[0]:]], axis=1)
    return tf.concat([upper_block, lower_block], axis=0)