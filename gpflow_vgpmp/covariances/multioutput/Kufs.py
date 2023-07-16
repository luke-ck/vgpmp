import tensorflow as tf
from gpflow.inducing_variables.multioutput.inducing_variables import SharedIndependentInducingVariables, SeparateIndependentInducingVariables
from gpflow.kernels.multioutput.kernels import SeparateIndependent, SharedIndependent
from gpflow_vgpmp.inducing_variables.inducing_variables import ConditionedVelocitySharedIndependentInducingVariables
from gpflow_vgpmp.covariances.Kufs import Kuf
from gpflow_vgpmp.kernel_conditioning.dispatch import K_conditioned
from gpflow_vgpmp.derivatives.dispatch import K_grad
from gpflow_vgpmp.kernels.kernels import FirstOrderKernelDerivativeSeparateIndependent, VanillaConditioningSeparateIndependent, VanillaConditioningSharedIndependent
from gpflow.base import TensorLike

@Kuf.register(SharedIndependentInducingVariables, FirstOrderKernelDerivativeSeparateIndependent, TensorLike)
def Kuf_fallback_shared_separate(
        inducing_variable: SharedIndependentInducingVariables,
        kernel: FirstOrderKernelDerivativeSeparateIndependent,
        Xnew: tf.Tensor,
):
	kuf = K_conditioned(inducing_variable.inducing_variable.Zy,
                         Xnew, kernel) # [L, M+2, P]
	ny_x_block = K_grad(inducing_variable.inducing_variable.ny, Xnew, kernel)  # [L, 2, P]
	return tf.concat([ny_x_block, kuf], axis=1)  # [L, M+4, P]

@Kuf.register(SharedIndependentInducingVariables, VanillaConditioningSeparateIndependent, TensorLike)
def Kuf_fallback_shared_separate(
        inducing_variable: SharedIndependentInducingVariables,
        kernel: VanillaConditioningSeparateIndependent,
        Xnew: tf.Tensor,
):
    kuf = K_conditioned(inducing_variable.inducing_variable.Zy,
                         Xnew, kernel)   # [L, 2, P]
    return kuf

@Kuf.register(SeparateIndependentInducingVariables, FirstOrderKernelDerivativeSeparateIndependent, TensorLike)
def Kuf_fallback_shared_separate(
        inducing_variable: SharedIndependentInducingVariables,
        kernel: FirstOrderKernelDerivativeSeparateIndependent,
        Xnew: tf.Tensor,
):
    kuf = K_conditioned(inducing_variable.inducing_variable.Zy,
                         Xnew, kernel)   # [L, M+2, P]
    ny_x_block = K_grad(inducing_variable.inducing_variable.ny, Xnew, kernel)  # [L, 2, P]
    return tf.concat([ny_x_block, kuf], axis=1)  # [L, M+4, P]

@Kuf.register(SeparateIndependentInducingVariables, VanillaConditioningSeparateIndependent, TensorLike)
def Kuf_fallback_shared_separate(
        inducing_variable: SharedIndependentInducingVariables,
        kernel: VanillaConditioningSeparateIndependent,
        Xnew: tf.Tensor,
):
    kuf = K_conditioned(inducing_variable.inducing_variable.Zy,
                         Xnew, kernel)   # [L, 2, P]
    return kuf

@Kuf.register(SharedIndependentInducingVariables, SharedIndependent, TensorLike)
def Kuf_fallback_shared_shared(
        inducing_variable: SharedIndependentInducingVariables,
        kernel: SharedIndependent,
        Xnew: tf.Tensor,
):
    return Kuf(inducing_variable.inducing_variable, kernel.kernel, Xnew)
