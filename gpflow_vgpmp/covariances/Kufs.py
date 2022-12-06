from gpflow.base import TensorLike
from gpflow.covariances import Kuf
from gpflow.kernels import Kernel

from ..inducing_variables.inducing_variables import VariableInducingPoints


@Kuf.register(VariableInducingPoints, Kernel, TensorLike)
def Kuf_kernel_variableinducingpoints(inducing_variable: VariableInducingPoints, kernel: Kernel, Xnew):
    return kernel(inducing_variable.Z, Xnew)  # k([u, y].T, f)
