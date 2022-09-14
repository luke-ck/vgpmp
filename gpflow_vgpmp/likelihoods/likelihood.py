import numpy as np
import tensorflow as tf
from gpflow.base import Parameter
from gpflow.config import default_float
from gpflow.likelihoods import GaussianMC
from gpflow.utilities import positive
from tensorflow_probability import bijectors as tfb

__all__ = ("likelihood")

TWOPI = np.log(2 * np.pi)


class VariationalMonteCarloLikelihood(GaussianMC):
    r"""
    Main class for computing likelihood. Interfaces with pybullet and the signed distance field.
    """

    def __init__(self, sigma_obs, sampler, sdf, alpha, radius, offset, joint_constraints, velocity_constraints,
                 DEFAULT_VARIANCE_LOWER_BOUND=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.variance = Parameter(sigma_obs, transform=positive(DEFAULT_VARIANCE_LOWER_BOUND))
        self.alpha = Parameter(alpha, transform=positive(DEFAULT_VARIANCE_LOWER_BOUND))
        self.sampler = sampler
        self.sdf = sdf
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

    @tf.function
    def _log_prob(self, F):
        r"""Takes in a tensor of joint configs, computes the 3D world position of sphere coordinates 
        and their signed distances for the respective joint configs, and returns the log likelihood for 
        those configurations

        Args:
            F (tf.Tensor): [S x N x D]

        Returns:
            [tf.Tensor]: [S x N]
        """

        L = self._sample_config_cost(F)
        logp = tf.reduce_sum(self._scalar_log_prob(L), axis=-1)
        return logp

    @tf.function
    def _scalar_log_prob(self, f):
        r"""

        Args:
            f (tf.Tensor): [S x N x P x 3]

        Returns:
            [tf.Tensor]: [S x N]
        """
        # tf.print(L)
        func = tf.range(f.shape[0])
        cost = tf.map_fn(  # this evaluates each [N x P x 3] of the S samples
            lambda i: self._hinge_loss(f[i]),
            func, fn_output_signature=default_float(), parallel_iterations=8,
        )

        S, N, P = cost.shape
        delta = tf.expand_dims(cost, -1)
        var = tf.eye(P, batch_shape=(S, N), dtype=default_float()) * self.variance

        res = tf.matmul(delta, tf.matmul(var, delta), transpose_a=True)
        dist_list = tf.reshape(res, shape=(S, N))

        return -0.5 * (TWOPI * P - tf.linalg.logdet(var) + dist_list)

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
            F (tf.Tensor): [S x N x D]

        Returns:
            [tf.Tensor]: [S x N x P x 3]
        """
        sample_dim = f.shape[1]
        func = tf.range(f.shape[0])
        return tf.map_fn(  # this evaluates each [N x D] of the S samples
            lambda i:
            self.sampler.sampleConfigs(f[i], sample_dim),
            func, fn_output_signature=(default_float())
        )

    @tf.custom_gradient
    def papirus_loss(self, data, epsilon=0.05):
        epsilon = tf.cast(epsilon, dtype=default_float())
        out = tf.where(data <= epsilon, - data + epsilon, 0.0)

        def grad(upstream):
            upstream = tf.where(data <= epsilon, tf.cast(-0.5, dtype=default_float()), upstream)
            return upstream, tf.cast(0., dtype=default_float())

        return out, grad

    @tf.function
    def _hinge_loss(self, data, epsilon=0.05):
        r"""
            Penalise configurations where arm is too close to objects with negative cost -d + \epsilon, otherwise 
        Args:
            data ([tf.Tensor]): [N x P x 3]
            epsilon (float, optional): Safety distance. Defaults to 0.5.

        Returns:
            [tf.Tensor]: [N x P]
        """
        # data = self.check_gradients(data)
        d = self._signed_distance_grad(data)
        return tf.where(d <= epsilon, - d + epsilon, 0.0)
        # return self.papirus_loss(d, epsilon)

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
            gradient = tf.stack([upstream + dist_grad[..., 0],
                                 upstream + dist_grad[..., 1],
                                 upstream + dist_grad[..., 2]], axis=-1)

            return gradient

        return dist, grad
