from abc import ABC

import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.base import Parameter, TensorData, Optional
from gpflow.base import default_float
from gpflow.inducing_variables import InducingVariables


class InducingPointsBase(InducingVariables, ABC):
    def __init__(self, Z: TensorData, dof, name: Optional[str] = None):
        """
        :param Z: the initial positions of the inducing points, size [M, D]
        """
        super().__init__(name=name)
        if not isinstance(Z, (tf.Variable, tfp.util.TransformedVariable)):
            Z = Parameter(Z)

        self._Z = Z
        self.start = tf.cast((tf.fill((1, dof), 0.)), dtype=default_float())
        self.end = tf.cast((tf.fill((1, dof), 1. * 100)), dtype=default_float())

    @property
    def Z(self):
        return self.Zy

    @property
    def Zy(self):
        return tf.concat([self.ny, self._Z], axis=0)

    @property
    def ny(self):
        return tf.concat([self.start, self.end], axis=0)

    @property
    def num_inducing(self) -> Optional[tf.Tensor]:
        return tf.shape(self._Z)[0]


class VariableInducingPoints(InducingPointsBase, ABC):
    def __init__(self, Z, dof, name=None):
        super().__init__(Z, dof, name=name)
