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
                 DEFAULT_VARIANCE_LOWER_BOUND=1e-14,
                 **kwargs):
        super().__init__(**kwargs)

        self.sdf = sdf
        self.sampler = sampler
        num_spheres_per_joint = robot.num_spheres_per_link
        sigma_obs_joints = tf.broadcast_to(sigma_obs, [robot.num_frames_for_spheres, 1])
        Sigma_obs = tf.reshape(tf.repeat(sigma_obs_joints, repeats=num_spheres_per_joint, axis=0), (1, robot.num_spheres))

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
        self._p = robot.num_spheres
        # self.k = Parameter(tf.constant(10, dtype=default_float()), transform=positive())

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
    def log_vel_prob(self, F, x):
        r"""
        Returns the log probability density log p(e|f) for samples S

        Args:
            F (tf.Tensor): [S x N x D]

        Returns:
            [tf.Tensor]: [S x N]
        """

        return self._log_vel_prob(F, x)

    @tf.function
    def _log_vel_prob(self, F, x):
        r"""
        Returns the log probability density log p(e|f) for samples S

        Args:
            F (tf.Tensor): [S x N x D]

        Returns:
            [tf.Tensor]: [S x N]
        """
        func = tf.range(F.shape[0])
        print(tf.gradients(F[0], x))
        grads = tf.map_fn(lambda i: tf.gradients(F[i], x)[0], func)
        return -0.5 * tf.reduce_sum(tf.square(grads - 1), axis=-1)


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
        # main_diag = 1 / self.variance
        # super_diag = tf.ones(self.variance.shape, dtype=tf.float64) * (1 / self.sigma_obs) * 0.9
        # # super_diag = super_diag[None, ...]
        # sub_diag = tf.ones(self.variance.shape, dtype=tf.float64) * (1 / self.sigma_obs) * 0.9
        # # sub_diag = sub_diag[None, ...]
        # main_diag = tf.reshape(tf.broadcast_to(main_diag, (cost.shape[0], 1, cost.shape[2])), (cost.shape[0], cost.shape[2]))
        # super_diag = tf.reshape(tf.broadcast_to(super_diag, (cost.shape[0], 1, cost.shape[2])), (cost.shape[0], cost.shape[2]))
        # sub_diag = tf.reshape(tf.broadcast_to(sub_diag, (cost.shape[0], 1, cost.shape[2])), (cost.shape[0], cost.shape[2]))
        # diags = [super_diag, main_diag, sub_diag]
        # rhs = tf.linalg.tridiagonal_matmul(diags, tf.transpose(cost, perm=(0, 2, 1)), diagonals_format='sequence')
        return - 0.5 * tf.reduce_sum(cost / self.variance[None, ...] * cost, axis=-1)
        # return -0.5 * tf.reduce_sum(cost * tf.transpose(rhs, perm=(0, 2, 1)), axis=-1)

    @tf.function
    def _sample_config_cost(self, f):
        r"""
        This function computes the cartesian position for each configuration
        in each sample. More explicitly, each iteration of map_fn returns a
        tensor of N x P x 3, where P is the dimensionality of spheres fitted
        on the robot arm. Under the hood we do forward kinematics, going from
        active joints to sphere locations in the cartesian space, and handle
        gradients associated with these operations.We do S iterations,
        one iteration per sample.
        Args:
            f (tf.Tensor): [S x N x D]
        Returns:
            [tf.Tensor]: [S x N x P x 3]
        """
        # Get the shape of the input tensor
        s, n, d = f.shape

        # Flatten the tensor so that it can be processed in one batch
        f = tf.reshape(f, (s * n, d))

        # Compute forward kinematics for all configurations in parallel using tf.vectorized_map
        k = tf.vectorized_map(self._fk_cost, f)

        # Reshape the output back to the original shape
        return tf.reshape(k, (s, n, self._p, 3))

    @tf.function
    def _fk_cost(self, joint_config):
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

        return tf.where(d <= self.epsilon, self.epsilon - d, 0.0)
        # return 1 / (1 + tf.math.exp(self.k * (d - self.epsilon))) * (- d + self.epsilon)

    @tf.custom_gradient
    def _signed_distance_grad(self, data):
        r"""
            Compute the signed distance from the sdf its gradient. Here we take the total gradient cost to be
            x_d * dx + y_d * dy + z_d * dz where d = [x_d, y_d, z_d]^T is the signed distance expressed in
            cartesian coordinates.
        Args:
            data([tf.Tensor]): [S x N x P x 3]
        Returns:
            d([tf.Tensor]): [S x N x P], grad([Tensor("gradients/...")]): [S x N x P x 3]
        """
        # norm_data = data
        norm_data = tf.math.subtract(data, self.offset)  # 'center' the data around the origin. This is because the
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
