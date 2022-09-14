import numpy as np
import pybullet as p
import tensorflow as tf
from gpflow.base import Parameter
from gpflow.base import default_float
from tensorflow_probability import bijectors as tfb


# <---------------- utilities ----------------->
def bounded_Z(low, high, Z):
    low, high = tf.cast(low, dtype=default_float()), tf.cast(high, dtype=default_float())
    """Make lengthscale tfp Parameter with optimization bounds."""
    sigmoid = tfb.Sigmoid(low, high)
    parameter = Parameter(Z, transform=sigmoid, dtype=default_float())
    return parameter


def initialize_Z(num_latent_gps, num_inducing):
    Z = tf.convert_to_tensor(np.array(
        [np.full(num_latent_gps, i) for i in np.linspace(0.1, 0.9, num_inducing)], dtype=np.float64))

    Z = bounded_Z(low=0.0 - 1e-2, high=1 + 1e-2, Z=Z)
    return Z


def bounded_lengthscale(low, high, lengthscale):
    """Make lengthscale tfp Parameter with optimization bounds."""
    affine = tfb.AffineScalar(shift=tf.cast(low, tf.float64),
                              scale=tf.cast(high - low, tf.float64))
    sigmoid = tfb.Sigmoid()
    logistic = tfb.Chain([affine, sigmoid])
    parameter = Parameter(lengthscale, transform=logistic, dtype=tf.float64)
    return parameter


def get_base(rotation, translation) -> np.array:
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = rotation
    px, py, pz = translation
    T = np.array([
        [r00, r01, r02, px],
        [r10, r11, r12, py],
        [r20, r21, r22, pz],
        [0, 0, 0, 1]
    ], dtype=np.float64)

    return T


def set_base(translation) -> np.array:
    assert len(translation) == 3
    rotation = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    T = get_base(rotation, translation)
    return T


def set_vec(translation) -> np.array:
    assert len(translation) == 3
    rotation = [0] * 9
    T = get_base(rotation, translation)
    return T


def get_world_transform(link, sphere):
    r""" Compute the world coordinates for each sphere by applying a translation from the link coordinate frame

    Args:
        link (tuple): link carthesian coordinates in world frame (0) and rotation quaternion (1)
        sphere (array): sphere offset from link frame in xyz

    Returns:
        (array): world coordinates of the sphere
    """
    return p.multiplyTransforms(link, [0, 0, 0, 1], sphere, [0, 0, 0, 1])


def get_transform_matrix(theta, d, a, alpha):
    """
    compute the homogenous transform matrix for a link given theta, d, a, alpha
    Args:
        theta (float): joint angle
        d (float): link length
        a (float): link offset from previous link
        alpha (float): link twist angle
    Returns:
        (array): homogenous transform matrix
    """
    T = np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
        [np.sin(theta), np.cos(theta) * np.cos(alpha), - np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ], dtype=np.float64)

    return T
