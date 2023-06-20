import abc
from abc import ABC

import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.base import Parameter, TensorData, Optional
from gpflow.base import default_float
from gpflow.inducing_variables import InducingVariables


class InducingPointsBase(InducingVariables, ABC):
    def __init__(self, Z: TensorData, dof: int, name: Optional[str] = None):
        """
        :param Z: the initial positions of the inducing points, size [M, D]
        """
        super().__init__(name=name)
        if not isinstance(Z, (tf.Variable, tfp.util.TransformedVariable)):
            Z = Parameter(Z)

        self._Z = Z
        self.start = tf.cast((tf.fill((1, dof), 0.)), dtype=default_float())
        self.end = tf.cast((tf.fill((1, dof), 1.)), dtype=default_float())

    @property
    def num_inducing(self) -> int:
        return self.__len__()

    def __len__(self) -> int:
        return tf.shape(self._Z)[0]


class InducingPointsInterface(InducingPointsBase, ABC):
    def __init__(self, Z, dof, name=None):
        super().__init__(Z, dof, name=name)

        self.len_ny = 2

    @property
    def Z(self):
        return tf.concat([self.ny, self._Z], axis=0)

    @property
    def ny(self):
        return tf.concat([self.start, self.end], axis=0)

    @property
    def dny(self):
        raise NotImplementedError

    @property
    def d2ny(self):
        raise NotImplementedError


class VariableInducingPoints(InducingPointsInterface, ABC):
    def __init__(self, Z: TensorData, dof, name: Optional[str] = None):
        """
        :param Z: the initial positions of the inducing points, size [M, D]
        """
        super().__init__(Z, dof, name=name)


class DerivativeInducingPoints(InducingPointsInterface, ABC):
    """
    Real-space inducing points
    """

    def __init__(self, Z: TensorData, dof, name: Optional[str] = None):
        """
        :param Z: the initial positions of the inducing points, size [M, D]
        """
        super().__init__(Z, dof, name=name)

        self.len_ny = 4

    @property
    def Z(self):
        return tf.concat([self.d2ny, self._Z], axis=0)

    @property
    def Zy(self):
        return tf.concat([self.dny, self._Z], axis=0)

    @property
    def dny(self):
        return tf.concat([self.start, self.end], axis=0)

    @property
    def d2ny(self):
        return tf.concat([self.start, self.end, self.start, self.end], axis=0)