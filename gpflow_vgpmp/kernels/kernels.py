from gpflow.kernels import SeparateIndependent, SharedIndependent


class FirstOrderKernelDerivativeSeparateIndependent(SeparateIndependent):
    def __init__(self, kernels, name=None):
        super().__init__(kernels, name)


class VanillaConditioningSeparateIndependent(SeparateIndependent):
    def __init__(self, kernels, name=None):
        super().__init__(kernels, name)


class VanillaConditioningSharedIndependent(SharedIndependent):
    def __init__(self, kernels, name=None):
        super().__init__(kernels, name)
