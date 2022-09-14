from typing import Union

import tensorflow as tf
from gpflow.covariances.dispatch import Kuu
from gpflow.inducing_variables.multioutput.inducing_variables import FallbackSharedIndependentInducingVariables
from gpflow.kernels.multioutput.kernels import (SeparateIndependent, IndependentLatent)

from gpflow_vgpmp.derivatives.dispatch import K_grad, K_grad_grad


# @Kuu.register(FallbackSharedIndependentInducingVariables, SharedIndependent)
# def Kuu_shared_shared(
#         inducing_variable: FallbackSharedIndependentInducingVariables,
#         kernel: SharedIndependent,
#         *,
#         jitter: float = 0.0,
# ) -> tf.Tensor:
#     # TODO
#     print(os.getcwd())
#     Kmm = Kuu(inducing_variable.inducing_variable, kernel.kernel)  # [M, M]
#     jittermat = tf.eye(inducing_variable.num_inducing, dtype=Kmm.dtype) * jitter
#     return Kmm + jittermat
#
#
# @Kuu.register(FallbackSeparateIndependentInducingVariables, SharedIndependent)
# def Kuu_fallback_separate_shared(
#         inducing_variable: FallbackSeparateIndependentInducingVariables,
#         kernel: SharedIndependent,
#         *,
#         jitter: float = 0.0,
# ) -> tf.Tensor:
#     # TODO
#     Kmm = tf.stack(
#         [Kuu(f, kernel.kernel) for f in inducing_variable.inducing_variable_list], axis=0
#     )  # [L, M, M]
#     jittermat = tf.eye(inducing_variable.num_inducing, dtype=Kmm.dtype)[None, :, :] * jitter
#     return Kmm + jittermat


@Kuu.register(FallbackSharedIndependentInducingVariables, (SeparateIndependent, IndependentLatent))
def Kuu_fallback_shared(
        inducing_variable: FallbackSharedIndependentInducingVariables,
        kernel: Union[SeparateIndependent, IndependentLatent],
        *,
        jitter: float = 0.0,
) -> tf.Tensor:
    Kmm = tf.stack(
        [Kuu(inducing_variable.inducing_variable, k) for idx, k in enumerate(kernel.kernels)], axis=0
    )  # [L, M+2, M+2]

    # Kzz = tf.pad(Kmm, [[0, 0], [2, 0], [2, 0]])
    zy_ny_block = K_grad(inducing_variable.inducing_variable.Zy,
                         inducing_variable.inducing_variable.ny, kernel)  # [L, M+2, 2]
    ny_zy_block = K_grad(inducing_variable.inducing_variable.ny,
                         inducing_variable.inducing_variable.Zy, kernel)  # [L, 2, M+2]

    ny_ny_block = K_grad_grad(inducing_variable.inducing_variable.ny, kernel)  # [L, 2, 2]
    down_block = tf.concat([zy_ny_block, Kmm], axis=-1)  # [L, M+2, M+4]
    up_block = tf.concat([ny_ny_block, ny_zy_block], axis=-1)  # [L, 2, M+4]
    Kzz = tf.concat([up_block, down_block], axis=1)  # [L, M+4, M+4]

    jittermat = tf.eye(inducing_variable.num_inducing + 4, dtype=Kmm.dtype)[None, :, :] * jitter
    return Kzz + jittermat
