from typing import List

import numpy as np
import pybullet as p
from gpflow.base import Parameter
from gpflow.base import default_float
from tensorflow_probability import bijectors as tfb
from scipy.spatial.transform import Rotation as rot
import tensorflow as tf
import time

# TODO: clean up this file
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

    Z = bounded_Z(low=0.09, high=0.91, Z=Z)
    return Z


def bounded_param(low, high, param):
    """Make a bounded tfp Parameter with optimization bounds."""
    affine = tfb.Shift(shift=tf.cast(low, tf.float64))(tfb.Scale(scale=tf.cast(high - low, tf.float64)))
    sigmoid = tfb.Sigmoid()
    logistic = tfb.Chain([affine, sigmoid])
    parameter = Parameter(param, transform=logistic, dtype=tf.float64)
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


def translation_vector(position):
    return np.concatenate([position, [1]]).reshape((4, 1))


def set_vec(translation) -> np.array:
    assert len(translation) == 3
    rotation = [0] * 9
    T = get_base(rotation, translation)
    return T


def quat_to_rotmat(quat: List) -> np.array:
    transform = rot.from_quat(quat)
    return transform.as_matrix()


def get_world_transform(link, sphere):
    r""" Compute the world coordinates for each sphere by applying a translation from the link coordinate frame

    Args:
        link (tuple): link carthesian coordinates in world frame (0) and rotation quaternion (1)
        sphere (array): sphere offset from link frame in xyz

    Returns:
        (array): world coordinates of the sphere
    """
    return p.multiplyTransforms(link, [0, 0, 0, 1], sphere, [0, 0, 0, 1])


def timing(f):
    def wrap(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        end = time.time()
        print('{:s} function took {:.3f} ms'.format(
            f.__name__, (end - start) * 1000.0))

        return ret

    return wrap
