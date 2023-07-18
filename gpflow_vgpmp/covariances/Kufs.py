from gpflow.base import TensorLike
from gpflow.covariances import Kuf
from gpflow.kernels import Kernel

from ..inducing_variables.inducing_variables import InducingPoints


@Kuf.register(InducingPoints, Kernel, TensorLike)
def _kuf_default_fallback(inducing_variable: InducingPoints, kernel: Kernel, Xnew):
    return kernel(inducing_variable.Z, Xnew)  # k([u, y].T, f)
