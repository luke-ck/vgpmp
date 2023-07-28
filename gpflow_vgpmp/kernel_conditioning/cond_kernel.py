from gpflow.kernels.stationaries import Kernel
from gpflow.base import TensorLike
from .dispatch import K_conditioned
from gpflow_vgpmp.inducing_variables.inducing_variables import ConditionedVariableInducingPoints


@K_conditioned.register(ConditionedVariableInducingPoints, ConditionedVariableInducingPoints, Kernel)
def k_cond_se_fallback(Z, X, kernel):
    K2 = kernel(Z.Zy, X)
    return K2


@K_conditioned.register(ConditionedVariableInducingPoints, TensorLike, Kernel)
def k_cond_se_fallback(Z, X, kernel):
    K2 = kernel(Z.Zy, X)
    return K2


@K_conditioned.register(TensorLike, TensorLike, Kernel)
def k_cond_se_fallback(Z, X, kernel):
    K2 = kernel(Z[..., None], X[..., None])
    return K2
