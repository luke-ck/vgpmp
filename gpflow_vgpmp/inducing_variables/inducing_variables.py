import abc
from abc import ABC

import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.base import Parameter, TensorData, Optional, TensorType
from gpflow.base import default_float
from gpflow.inducing_variables import InducingVariables
from gpflow.inducing_variables.multioutput import SharedIndependentInducingVariables, SeparateIndependentInducingVariables

class ConditionedSharedIndependentInducingVariables(SharedIndependentInducingVariables):
    def __init__(self, inducing_variable: TensorType):
        super().__init__(inducing_variable)

class ConditionedSeparateIndependentInducingVariables(SeparateIndependentInducingVariables):
    def __init__(self, inducing_variable: TensorType):
        super().__init__(inducing_variable)

class ConditionedVelocitySharedIndependentInducingVariables(SharedIndependentInducingVariables):
    def __init__(self, inducing_variable: TensorType):
        super().__init__(inducing_variable)

class InducingPointsBase(InducingVariables, ABC):
    def __init__(self, Z: TensorData, conditioned_timesteps, name: Optional[str] = None):
        """
        :param Z: the initial positions of the inducing points, size [M, D]
        """
        super().__init__(name=name)
        if not isinstance(Z, (tf.Variable, tfp.util.TransformedVariable)):
            Z = Parameter(Z)

        self._Z = Z
        self.conditioned_timesteps = tf.cast(conditioned_timesteps, dtype=default_float())
        assert self._Z.shape[1] == self.conditioned_timesteps.shape[1], "The number of degrees of freedom of the trainable inducing points and the conditioned timesteps must be the same. Right now it is {} and {}, respectively.".format(self._Z.shape[1], self.conditioned_timesteps.shape[1])

    @property
    def num_inducing(self) -> int:
        return self.__len__()

    def __len__(self) -> int:
        return tf.shape(self._Z)[0]


class InducingPointsInterface(InducingPointsBase, ABC):
    def __init__(self, Z, conditioned_timesteps, name=None):
        super().__init__(Z, conditioned_timesteps, name=name)

        self.len_ny = 2

    @property
    def Z(self):
        return tf.concat([self.ny, self._Z], axis=0)

    @property
    def ny(self):
        return tf.concat([self.conditioned_timesteps], axis=0)

    @property
    def dny(self):
        raise NotImplementedError

    @property
    def d2ny(self):
        raise NotImplementedError


class ConditionedVariableInducingPoints(InducingPointsInterface, ABC):
    def __init__(self, Z: TensorData, conditioned_timesteps, name: Optional[str] = None):
        """
        :param Z: the initial positions of the inducing points, size [M, D]
        """
        super().__init__(Z, conditioned_timesteps, name=name)

    @property
    def Zy(self):
        return tf.concat([self.ny, self._Z], axis=0)


class FirstOrderDerivativeInducingPoints(InducingPointsInterface, ABC):
    """
    Real-space inducing points
    """

    def __init__(self, Z: TensorData, conditioned_timesteps, name: Optional[str] = None):
        """
        :param Z: the initial positions of the inducing points, size [M, D]
        """
        super().__init__(Z, conditioned_timesteps, name=name)

        self.len_ny = conditioned_timesteps.shape[0]

    @property
    def Z(self):
        return tf.concat([self.d2ny, self._Z], axis=0)

    @property
    def Zy(self):
        return tf.concat([self.dny, self._Z], axis=0)

    @property
    def dny(self):
        return tf.concat([self.conditioned_timesteps], axis=0)
      
    @property
    def d2ny(self):
        return tf.concat([self.conditioned_timesteps, self.conditioned_timesteps], axis=0)

