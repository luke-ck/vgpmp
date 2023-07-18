from typing import Union

import tensorflow as tf
from gpflow.covariances.dispatch import Kuu
from gpflow.inducing_variables.multioutput.inducing_variables import FallbackSharedIndependentInducingVariables, FallbackSeparateIndependentInducingVariables
from gpflow.kernels.multioutput.kernels import (SeparateIndependent, IndependentLatent, SharedIndependent)
from gpflow_vgpmp.inducing_variables.inducing_variables import (ConditionedSeparateIndependentInducingVariables, ConditionedSharedIndependentInducingVariables, ConditionedVelocitySharedIndependentInducingVariables)
from gpflow_vgpmp.kernel_conditioning.dispatch import K_conditioned
from gpflow_vgpmp.derivatives.dispatch import K_grad, K_grad_grad
from gpflow_vgpmp.kernels.kernels import FirstOrderKernelDerivativeSeparateIndependent, VanillaConditioningSeparateIndependent


@Kuu.register(FallbackSharedIndependentInducingVariables, (FirstOrderKernelDerivativeSeparateIndependent, IndependentLatent))
def Kuu_fallback_shared_separate(
        inducing_variable: ConditionedSharedIndependentInducingVariables,
        kernel: Union[FirstOrderKernelDerivativeSeparateIndependent, IndependentLatent],
        *,
        jitter: float = 0.0,
) -> tf.Tensor:
    Kmm = K_conditioned(inducing_variable.inducing_variable.Zy,
                         inducing_variable.inducing_variable.Zy, kernel)
    zy_ny_block = K_grad(inducing_variable.inducing_variable.Zy,
                         inducing_variable.inducing_variable.ny, kernel)  # [L, M+2, 2]
    ny_zy_block = K_grad(inducing_variable.inducing_variable.ny,
                         inducing_variable.inducing_variable.Zy, kernel)  # [L, 2, M+2]

    ny_ny_block = K_grad_grad(inducing_variable.inducing_variable.ny, kernel)  # [L, 2, 2]
    Kmm = tf.concat([zy_ny_block, Kmm], axis=-1)  # [L, M+2, M+4] (down block)
    up_block = tf.concat([ny_ny_block, ny_zy_block], axis=-1)  # [L, 2, M+4]
    Kmm = tf.concat([up_block, Kmm], axis=1)  # [L, M+4, M+4]

    padding = inducing_variable.inducing_variable.ny.shape[0] + inducing_variable.inducing_variable.dny.shape[0]
    jittermat = tf.eye(inducing_variable.num_inducing + padding , dtype=Kmm.dtype)[None, :, :] * jitter
    return Kmm + jittermat

@Kuu.register(FallbackSharedIndependentInducingVariables, (VanillaConditioningSeparateIndependent, IndependentLatent))
def Kuu_fallback_shared_separate(
        inducing_variable: FallbackSharedIndependentInducingVariables,
        kernel: Union[VanillaConditioningSeparateIndependent, IndependentLatent],
        *,
        jitter: float = 0.0,
) -> tf.Tensor:

    Kmm = K_conditioned(inducing_variable.inducing_variable.Zy,
                         inducing_variable.inducing_variable.Zy, kernel)
    jittermat = tf.eye(inducing_variable.num_inducing + inducing_variable.inducing_variable.ny.shape[0], dtype=Kmm.dtype)[None, :, :] * jitter
    return Kmm + jittermat

@Kuu.register(FallbackSeparateIndependentInducingVariables, (FirstOrderKernelDerivativeSeparateIndependent, IndependentLatent))
def Kuu_fallback_shared_separate(
        inducing_variable: ConditionedSharedIndependentInducingVariables,
        kernel: Union[FirstOrderKernelDerivativeSeparateIndependent, IndependentLatent],
        *,
        jitter: float = 0.0,
) -> tf.Tensor:
    Kmm = K_conditioned(inducing_variable.inducing_variable.Zy,
                         inducing_variable.inducing_variable.Zy, kernel)
    zy_ny_block = K_grad(inducing_variable.inducing_variable.Zy,
                         inducing_variable.inducing_variable.ny, kernel)  # [L, M+2, 2]
    ny_zy_block = K_grad(inducing_variable.inducing_variable.ny,
                         inducing_variable.inducing_variable.Zy, kernel)  # [L, 2, M+2]

    ny_ny_block = K_grad_grad(inducing_variable.inducing_variable.ny, kernel)  # [L, 2, 2]
    Kmm = tf.concat([zy_ny_block, Kmm], axis=-1)  # [L, M+2, M+4] (down block)
    up_block = tf.concat([ny_ny_block, ny_zy_block], axis=-1)  # [L, 2, M+4]
    Kmm = tf.concat([up_block, Kmm], axis=1)  # [L, M+4, M+4]

    padding = inducing_variable.inducing_variable.ny.shape[0] + inducing_variable.inducing_variable.dny.shape[0]
    jittermat = tf.eye(inducing_variable.num_inducing + padding , dtype=Kmm.dtype)[None, :, :] * jitter
    return Kmm + jittermat

@Kuu.register(FallbackSeparateIndependentInducingVariables, (VanillaConditioningSeparateIndependent, IndependentLatent))
def Kuu_fallback_shared_separate(
        inducing_variable: FallbackSharedIndependentInducingVariables,
        kernel: Union[VanillaConditioningSeparateIndependent, IndependentLatent],
        *,
        jitter: float = 0.0,
) -> tf.Tensor:

    Kmm = K_conditioned(inducing_variable.inducing_variable.Zy,
                         inducing_variable.inducing_variable.Zy, kernel)
    jittermat = tf.eye(inducing_variable.num_inducing + inducing_variable.inducing_variable.ny.shape[0], dtype=Kmm.dtype)[None, :, :] * jitter
    return Kmm + jittermat

@Kuu.register(FallbackSharedIndependentInducingVariables, SharedIndependent)
def Kuu_fallback_shared_shared(
        inducing_variable: FallbackSharedIndependentInducingVariables,
        kernel: Union[SharedIndependent, IndependentLatent],
        *,
        jitter: float = 0.0,
) -> tf.Tensor:
    Kmm = Kuu(inducing_variable.inducing_variable, kernel.kernel)

    jittermat = tf.eye(inducing_variable.num_inducing + inducing_variable.inducing_variable.ny.shape[0], dtype=Kmm.dtype) * jitter
    return Kmm + jittermat

