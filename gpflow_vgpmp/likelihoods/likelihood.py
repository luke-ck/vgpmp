from abc import ABC
import numpy as np
import tensorflow as tf
from gpflow.base import Parameter
from gpflow.config import default_float
from gpflow.likelihoods import GaussianMC, Gaussian
from gpflow.utilities import positive
from gpflow_vgpmp.utils.ops import bounded_param
from tensorflow_probability import bijectors as tfb

__all__ = "likelihood"

TWOPI = np.log(2 * np.pi)


class VariationalMonteCarloLikelihood(Gaussian, ABC):
    r"""
    Main class for computing likelihood. Interfaces with pybullet and the signed distance field.
    """

    def __init__(self, sigma_obs, num_spheres, sampler, sdf, radius, offset, joint_constraints,
                 velocity_constraints, train_sigma, no_frames_for_spheres, epsilon,
                 DEFAULT_VARIANCE_LOWER_BOUND=1e-14, **kwargs):
        super().__init__(**kwargs)

        self.sampler = sampler
        self.sdf = sdf
        # sigma_obs_joints = decay_sigma(sigma_obs, num_latent_gps, 1.5)
        sigma_obs_joints = tf.broadcast_to(sigma_obs, [no_frames_for_spheres, 1])
        Sigma_obs = tf.reshape(tf.repeat(sigma_obs_joints, repeats=self.sampler.num_spheres, axis=0), (1, num_spheres))
        self.variance = Parameter(Sigma_obs, transform=positive(DEFAULT_VARIANCE_LOWER_BOUND), trainable=train_sigma)
        self.offset = tf.constant(offset, dtype=default_float(), shape=(1, 3))
        self.radius = tf.constant(radius, dtype=default_float(), shape=(1, len(radius)))
        self.joint_constraints = tf.constant(joint_constraints, shape=(len(joint_constraints) // 2, 2),
                                             dtype=default_float())
        self.velocity_constraints = tf.constant(velocity_constraints, shape=(len(velocity_constraints) // 2, 2),
                                                dtype=default_float())
        self.joint_sigmoid = tfb.Sigmoid(
            low=self.joint_constraints[:, 1],
            high=self.joint_constraints[:, 0],
        )
        self.vel_sigmoid = tfb.Sigmoid(
            low=self.velocity_constraints[:, 1],
            high=self.velocity_constraints[:, 0]
        )
        self.normal = tf.constant([0., 0., 1.], dtype=default_float(), shape=(1, 1, 1, 3))
        self.epsilon = tf.constant(epsilon, dtype=default_float())
        self._p = num_spheres

    def decay_sigma(sigma_obs, num_latent_gps, decay_rate):
        func = tf.range(num_latent_gps + 1)
        return tf.map_fn(lambda i: sigma_obs / (decay_rate * tf.cast(i + 1, dtype=default_float())), func,
                        fn_output_signature=default_float())

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

    def _log_prob(self, F):
        r"""
        Takes in a tensor of joint configs, computes the 3D world position of sphere coordinates
        and their signed distances for the respective joint configs, and returns the log likelihood for
        those configurations. Ideally shape checks should be done as well.
        Args:
            F (tf.Tensor): [S x N x D]
        Returns:
            logp (tf.Tensor): [S x N]
        """
        L = self._sample_config_cost(F)  # S x N x P x 3
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
        return - 0.5 * tf.reduce_sum(cost / self.variance[None, ...] * cost, axis=-1)

    @tf.function
    def _sample_config_cost(self, f):
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
        return self.sampler._fk_cost(tf.expand_dims(joint_config, axis=1))

    @tf.function
    def _hinge_loss(self, data):
        r"""
            Penalise configurations where arm is too close to objects with negative cost -d + \epsilon
        Args:
            data ([tf.Tensor]): [N x P x 3]
            epsilon (float, optional): Safety distance. Defaults to 0.5.

        Returns:
            [tf.Tensor]: [N x P]
        """
        d = self._signed_distance_grad(data)
        loss1 = tf.where(d <= self.epsilon, - d + self.epsilon, 0.0)
        return loss1
        # return self.smoothed_hinge_loss(d, self.epsilon)

    @tf.function
    def smoothed_hinge_loss(self, data, epsilon):
        out = tf.where(tf.math.logical_and(data > tf.cast(0.0, dtype=default_float()), data < epsilon),
                       (epsilon - data) ** 2 / 2, data)
        out = tf.where(out > epsilon, tf.cast(0.0, dtype=default_float()), out)
        out = tf.where(out < tf.cast(0.0, dtype=default_float()), tf.cast(0.5, dtype=default_float()) - out, out)
        return out

    @tf.custom_gradient
    def _signed_distance_grad(self, data):
        r"""
            Compute the signed distance from the sdf its gradient. Here we take the total gradient cost to be
            x_d * dx + y_d * dy + z_d * dz where d = [x_d, y_d, z_d]^T is the signed distance expressed in
            cartesian coordinates.
        Args:
            data([tf.Tensor]): [N x P x 3]
        Returns:
            d([tf.Tensor]): [N x P], grad([Tensor("gradients/...")]): [N x P x 3]
        """
        # norm_data = data
        norm_data = tf.math.subtract(data, self.offset)  # 'center' the data around the origin. This is because the
        # sdf is centered around the origin.
        dist = self.sdf.get_distance_tf(norm_data)
        dist = tf.math.subtract(dist, self.radius)  # subtract radius from distance from the sphere centre to the
        # object body
        dist_grad = self.sdf.get_distance_grad_tf(norm_data)

        def grad(upstream):
            gradient = tf.stack([upstream * dist_grad[..., 0],
                                 upstream * dist_grad[..., 1],
                                 upstream * dist_grad[..., 2]], axis=-1)

            return gradient

        return dist, grad
