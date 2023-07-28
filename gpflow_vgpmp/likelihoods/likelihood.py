from abc import ABC
from typing import List

import tensorflow as tf
from gpflow.base import Parameter
from gpflow.config import default_float
from gpflow.likelihoods import Gaussian
from gpflow.utilities import positive
from tensorflow_probability import bijectors as tfb

__all__ = "likelihood"

from gpflow_vgpmp.utils.robot import Robot
from gpflow_vgpmp.utils.sampler import Sampler
from gpflow_vgpmp.utils.sdf_utils import SignedDistanceField


class VariationalMonteCarloLikelihood(Gaussian, ABC):
    r"""
    Main class for computing likelihood. Interfaces with pybullet and the signed distance field.
    """

    def __init__(self,
                 sigma_obs: float,
                 robot: Robot,
                 sampler: Sampler,
                 sdf: SignedDistanceField,
                 offset: List[float],
                 epsilon: float = 0.05,
                 DEFAULT_VARIANCE_LOWER_BOUND=1e-5,
                 **kwargs):
        super().__init__(**kwargs)

        self.sdf = sdf
        self.sampler = sampler
        num_spheres_per_joint = robot.num_spheres_per_link
        sigma_obs_joints = tf.broadcast_to(sigma_obs, [robot.num_frames_for_spheres, 1])
        Sigma_obs = tf.reshape(tf.repeat(sigma_obs_joints, repeats=num_spheres_per_joint, axis=0),
                               (1, robot.num_spheres))
        # Sigma_obs = tf.math.log(1 / Sigma_obs)
        self.variance = Parameter(Sigma_obs, transform=positive(DEFAULT_VARIANCE_LOWER_BOUND))

        self.offset = tf.constant(offset, dtype=default_float(), shape=(1, 3))
        self.sphere_radii = tf.constant(robot.sphere_radii, dtype=default_float(), shape=(1, len(robot.sphere_radii)))
        self.joint_constraints = tf.constant(robot.joint_limits, shape=(len(robot.joint_limits) // 2, 2),
                                             dtype=default_float())
        self.velocity_constraints = tf.constant(robot.velocity_limits, shape=(len(robot.velocity_limits) // 2, 2),
                                                dtype=default_float())
        self.joint_sigmoid = tfb.Sigmoid(
            low=self.joint_constraints[:, 1],
            high=self.joint_constraints[:, 0],
        )
        self.epsilon = tf.constant(epsilon, dtype=default_float())
        self.p = robot.num_spheres
        self.k = Parameter(tf.constant(2, dtype=default_float()), transform=positive())

    @tf.function
    def log_prob(self, F):
        r"""
        Returns the log probability density log p(e|f) for samples S

        Args:
            F (tf.Tensor): [S x N x D]

        Returns:
            [tf.Tensor]: [S x N]
        """

        return self._log_prob(F)

    @tf.function
    def _log_prob(self, f):
        r"""
        Takes in a tensor of joint configs, computes the 3D world position of sphere coordinates
        and their signed distances for the respective joint configs, and returns the log likelihood for
        those configurations. Ideally shape checks should be done as well.
        Args:
            f (tf.Tensor): [S x N x D]
        Returns:
            logp (tf.Tensor): [S x N]
        """
        L = self._sample_config_cost(f)  # S x N x P x 3
        logp = self._scalar_log_prob(L)
        return logp

    @tf.function
    def _scalar_log_prob(self, f):
        r"""
        Args:
            f (tf.Tensor): [S x N x P x 3]
        Returns:
            [tf.Tensor]: [S x N]
        """

        cost = self._hinge_loss(f)

        # reg = tf.reduce_sum(tf.abs(cost), axis=-1)

        return - 0.5 * tf.reduce_sum(cost / self.variance[None, ...] * cost, axis=-1) #- self.k * reg

    @tf.function
    def _sample_config_cost(self, f):
        r"""
        This function computes the cartesian position for each configuration
        in each sample in parallel via vectorization.
        More explicitly, each iteration of _compute_forward_kinematics_cost returns a
        tensor of N x P x 3, where P is the dimensionality of spheres fitted
        on the robot arm. Under the hood we do forward kinematics, going from
        active joints to sphere locations in the cartesian space, and handle
        gradients associated with these operations.
        Args:
            f (tf.Tensor): [S x N x D]
        Returns:
            [tf.Tensor]: [S x N x P x 3]
        """
        # Get the shape of the input tensor
        s, n, d = f.shape

        # Flatten the tensor so that it can be processed in one batch and compute batch forward kinematics
        f = tf.reshape(f, (s * n, d))

        k = tf.vectorized_map(self._compute_forward_kinematics_cost, f)

        # Reshape the output back to the original shape
        return tf.reshape(k, (s, n, self.p, 3))

    @tf.function
    def _compute_forward_kinematics_cost(self, joint_config):
        return self.sampler.forward_kinematics_cost(tf.expand_dims(joint_config, axis=1))

    @tf.function
    def _hinge_loss(self, data):
        r"""
            Penalise configurations where arm is too close to objects with negative cost -d + \epsilon
        Args:
            data ([tf.Tensor]): [N x P x 3]
        Returns:
            [tf.Tensor]: [N x P]
        """
        d = self._signed_distance_grad(data)

        # d = tf.where(d <= self.epsilon, self.epsilon - d, 0.0)
        return tf.math.maximum(self.epsilon - d, 0.0)
        # return 1 / (1 + tf.math.exp(self.k * (d - self.epsilon))) * (- d + self.epsilon)

    @tf.custom_gradient
    def _signed_distance_grad(self, data):
        r"""
            Compute the signed distance from the sdf its gradient. Here we take the total gradient cost to be
            x_d * dx + y_d * dy + z_d * dz where d = [x_d, y_d, z_d]^T is the signed distance expressed in
            cartesian coordinates. Currently, we only support environments which are shifted from the origin,
            so either extend this function to account for rotation or change the pose of the environment before
            generating the sdf.
        Args:
            data([tf.Tensor]): [S x N x P x 3]
        Returns:
            d([tf.Tensor]): [S x N x P], grad([Tensor("gradients/...")]): [S x N x P x 3]
        """
        # norm_data = data
        norm_data = tf.math.subtract(data, self.offset)  # 'center' the data around the origin. This is because the
        # TODO: this should be extended to support rotation of the sdf
        # sdf is centered around the origin.
        dist = self.sdf.get_distance_tf(norm_data)

        dist = tf.math.subtract(dist, self.sphere_radii)  # subtract radius from distance from the sphere centre to the
        # object body
        dist_grad = self.sdf.get_distance_grad_tf(norm_data)

        def grad(upstream):
            gradient = tf.stack([upstream * dist_grad[..., 0],
                                 upstream * dist_grad[..., 1],
                                 upstream * dist_grad[..., 2]], axis=-1)

            return gradient

        return dist, grad
