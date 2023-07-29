import tensorflow as tf
from gpflow.kernels import SquaredExponential, Matern52
from gpflow_vgpmp.derivatives.dispatch import K_grad, K_grad_grad


# Assuming se_kernel and k_grad_se_fallback, k_grad_grad_se_fallback are defined elsewhere

def test_first_order_gradient():
    lengthscale = 2.0
    variance = 0.8
    x = tf.constant([1.0, 2.0, 3.0], dtype=tf.float64)
    y = tf.constant([2.0, 3.0, 4.0], dtype=tf.float64)
    kern = Matern52(lengthscales=lengthscale, variance=variance)

    jacobian_autodiff = []
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            with tf.GradientTape() as tape:
                tape.watch([x, y])
                k = kern(x[i:i + 1], y[j:j + 1])
            grad = tape.gradient(k, y)
            jacobian_autodiff.append(tf.reduce_sum(grad))

    jacobian_autodiff = tf.reshape(jacobian_autodiff, (x.shape[0], y.shape[0]))
    jacobian_manual = K_grad(x, y, kern)
    # Verify that the two matrices are close to each other
    tf.debugging.assert_near(jacobian_autodiff, jacobian_manual, rtol=1e-5)


def test_second_order_gradient():
    lengthscale = 1.5
    variance = 0.5
    x = tf.constant([1.0, 2.0, 3.0], dtype=tf.float64)
    y = tf.constant([1.0, 2.0, 3.0],
                    dtype=tf.float64) + 1e-5  # This tolerance is needed because autodiff does not handle division by zero really well.
    # The division by zero is caused from the Matern 5/2 derivatives.
    kern = SquaredExponential(lengthscales=lengthscale, variance=variance)

    hessian_autodiff = []
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            with tf.GradientTape(persistent=True) as outer_tape:
                outer_tape.watch([x, y])
                with tf.GradientTape(persistent=True) as inner_tape:
                    inner_tape.watch([x, y])
                    k = kern(x[i:i + 1], y[j:j + 1])
                grad_x = inner_tape.gradient(k, x)
            hessian_xy = outer_tape.gradient(grad_x, y)
            hessian_autodiff.append(tf.reduce_sum(hessian_xy))

    hessian_autodiff = tf.reshape(hessian_autodiff, (x.shape[0], y.shape[0]))
    hessian_manual = K_grad_grad(x, y, kern)
    # Verify that the two matrices are close to each other
    tf.debugging.assert_near(hessian_autodiff, hessian_manual, rtol=1e-5)
