from gpflow.base import TensorLike
from gpflow.covariances import Kuf
from gpflow.kernels import Kernel

from ..inducing_variables.inducing_variables import InducingPointsInterface


@Kuf.register(InducingPointsInterface, Kernel, TensorLike)
def Kuf_kernel_variableinducingpoints(inducing_variable: InducingPointsInterface, kernel: Kernel, Xnew):
    return kernel(inducing_variable.Zy, Xnew)  # k([u, y].T, f)
