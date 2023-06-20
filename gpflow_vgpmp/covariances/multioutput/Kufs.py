import tensorflow as tf
from gpflow.inducing_variables.multioutput.inducing_variables import SharedIndependentInducingVariables
from gpflow.kernels.multioutput.kernels import SeparateIndependent, SharedIndependent

from gpflow_vgpmp.covariances.Kufs import Kuf
from gpflow_vgpmp.derivatives.dispatch import K_grad


@Kuf.register(SharedIndependentInducingVariables, SeparateIndependent, object)
def Kuf_fallback_shared_separate(
        inducing_variable: SharedIndependentInducingVariables,
        kernel: SeparateIndependent,
        Xnew: tf.Tensor,
):
    kuf = tf.stack(
        [Kuf(inducing_variable.inducing_variable, k, Xnew) for k in kernel.kernels], axis=0
    )  # [L, M+2, P]
    ny_x_block = K_grad(inducing_variable.inducing_variable.ny, Xnew, kernel)  # [L, 2, P]
    # kuf = tf.pad(kuf, [[0, 0], [2, 0], [0, 0]])
    return tf.concat([ny_x_block, kuf], axis=1)  # [L, M+4, P]
    # return kuf


@Kuf.register(SharedIndependentInducingVariables, SharedIndependent, object)
def Kuf_fallback_shared_shared(
        inducing_variable: SharedIndependentInducingVariables,
        kernel: SharedIndependent,
        Xnew: tf.Tensor,
):
    return Kuf(inducing_variable.inducing_variable, kernel.kernel, Xnew)
