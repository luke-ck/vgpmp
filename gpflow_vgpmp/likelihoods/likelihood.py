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
        self.ee_position_pringles = tf.constant([-0.00472223, 0.28439094, 0.98321232], dtype =default_float())
        self.ee_orientation_pringles = tf.reshape(tf.constant([[-0.04832382, 0.9886533, -0.1422303],
                                        [-0.02910649, 0.14094235, 0.98958985],
                                        [ 0.99840754, 0.05196058, 0.02196535]], dtype =default_float()), shape=[9])

        self.robot_pos_pringle = tf.constant([[-8.00000000e-02,  0.00000000e+00,  2.00000000e-02],
                                              [ 0.00000000e+00,  0.00000000e+00,  4.00000000e-02],
                                              [ 4.66474749e-02, -3.77360979e-02,  3.33000000e-01],
                                              [ 1.55491583e-02, -1.25786993e-02,  2.53000000e-01],
                                              [ 0.00000000e+00,  0.00000000e+00,  1.73000000e-01],
                                              [-3.88728958e-02,  3.14467482e-02,  3.32999816e-01],
                                              [-3.52279803e-02, -1.17468452e-02,  4.06627448e-01],
                                              [-3.93576440e-02, -4.86510891e-02,  4.80255042e-01],
                                              [-5.31322026e-02, -6.56794693e-02,  5.31794307e-01],
                                              [-6.14994132e-02, -6.85744924e-02,  5.72063509e-01],
                                              [-8.51143636e-02, -1.75342917e-02,  6.47808027e-01],
                                              [-1.32522015e-01, -9.62383668e-03,  6.32318054e-01],
                                              [-7.58764091e-02,  3.97528213e-02,  7.77608388e-01],
                                              [-8.00923518e-02, -1.79209212e-03,  7.49657610e-01],
                                              [-5.25372741e-02, -2.50975762e-02,  6.59393707e-01],
                                              [-6.13766484e-04, -3.50165473e-02,  6.76143733e-01],
                                              [-8.00060802e-02,  2.74899922e-01,  9.23326803e-01],
                                              [-1.23927713e-01,  2.68078969e-01,  9.46225815e-01],
                                              [-1.29465490e-01,  2.24370566e-01,  9.22584569e-01],
                                              [-1.29503548e-01,  1.90085938e-01,  9.01381671e-01],
                                              [-1.29541607e-01,  1.55801311e-01,  8.80178773e-01],
                                              [-1.08726404e-01,  1.16185480e-01,  8.42798119e-01],
                                              [-7.03425475e-02,  7.92980299e-02,  7.96257861e-01],
                                              [-3.60844513e-02,  2.81720843e-01,  9.00427773e-01],
                                              [-4.10510272e-03,  2.54336481e-01,  9.75443892e-01],
                                              [-8.07661957e-03,  3.14170340e-01,  9.85648613e-01],
                                              [-1.61006527e-02,  3.63558130e-01,  9.84969551e-01],
                                              [ 2.62141740e-02,  3.68590744e-01,  1.03223612e+00],
                                              [-4.16673123e-02,  5.06343946e-01,  9.87894724e-01],
                                              [-4.94203909e-02,  4.90090675e-01,  9.87045638e-01],
                                              [-4.65757849e-02,  4.70298878e-01,  9.86606331e-01],
                                              [-3.17807793e-02,  5.07753370e-01,  9.88414330e-01],
                                              [-1.97607918e-02,  4.94318946e-01,  9.88604456e-01],
                                              [-1.69161859e-02,  4.74527149e-01,  9.88165149e-01],
                                              [-8.35269778e-02,  4.11508210e-01,  9.83103941e-01],
                                              [-2.42077796e-02,  4.19964751e-01,  9.86221576e-01],
                                              [ 3.51114185e-02,  4.28421292e-01,  9.89339211e-01]], dtype=default_float())

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
        L, matrices = self._sample_config_cost(F)  # S x N x P x 3
        logp = self._scalar_log_prob(L)
        force_pos, force_orientation  = self._grasp_log_prob(matrices)
        return logp, force_pos, force_orientation

    @tf.function
    def _grasp_log_prob(self, f):
        r"""
        Returns the log probability density log p(e|f) for samples S

        Args:
            F (tf.Tensor): [S x N x D x 3]

        Returns:
            [tf.Tensor]: [S]
        """
        
        ee_position = tf.reshape(f[:, -1, :3, 3], [f.shape[0], 1, 3])
        ee_orientation = f[:, -20:, :3, :3]
        ee_orientation = tf.reshape(ee_orientation, [f.shape[0], 20, 9])
        # tf.print(f[0, 0, ...])
        # tf.print(f.shape)
        # # print(tf.squeeze(matrices, axis=[0, 1]))
        # tf.print(f[0, 0, :3, :3])
        # tf.print(f[0, 0, :3, 3])
        # spheres = f[:, -1, 24, :]
        # spheres = f[:, -1, :, :]
        
        position_mean = ee_position - self.ee_position_pringles[None, None, ...]
        orientation_mean = ee_orientation - self.ee_orientation_pringles[None, None, ...]
        # mean = spheres - self.robot_pos_pringle[None, ...]
        
        force_position = tf.math.reduce_sum(position_mean * position_mean, axis=-1) / 0.00000005
        force_orientation = tf.math.reduce_sum(orientation_mean * orientation_mean, axis=-1) / 0.00005
        return - 0.5 * force_position, - 0.5 * force_orientation

    @tf.function
    def log_normalization_constant(self, sigma, k=0.2):
        # pi = tf.ones_like(sigma, dtype=tf.float64)
        sigma_inv_sqrt = tf.math.rsqrt(sigma)
        erf_arg = tf.cast(tf.math.sqrt(1/2), dtype=default_float()) * sigma_inv_sqrt * tf.cast(k, dtype=default_float())
        log_C = tf.math.log(2 * sigma * np.pi) + tf.math.log(tf.math.erf(erf_arg) - tf.math.erf(tf.cast(0.0, dtype=default_float())))
        return tf.reduce_sum(log_C)


    @tf.function
    def _scalar_log_prob(self, f):
        r"""
        Args:
            f (tf.Tensor): [S x N x P x 3]

        Returns:
            [tf.Tensor]: [S x N]
        """
        cost = self._hinge_loss(f) #tf.clip_by_value(self._hinge_loss(f), clip_value_min=0, clip_value_max=0.75)
        S, N, P = cost.shape
        delta = tf.expand_dims(cost, -1)
        var = tf.eye(P, batch_shape=(S, N), dtype=default_float()) / self.variance

        res = tf.matmul(delta, tf.matmul(var, delta), transpose_a=True)
        dist_list = tf.reshape(res, shape=(S, N))

        # normal_cost = tf.reshape(f[:, :, 26, :], [f.shape[0], f.shape[1], 1, f.shape[3]]) - f[:, :, 27:, :]
        # normal_cost = tf.linalg.normalize(normal_cost, axis=-1)[0]
        # normal_cost = tf.reduce_sum(normal_cost * self.normal, axis=-1)
        # new_delta = tf.expand_dims(normal_cost, -1)
        # normal_cost = tf.matmul(new_delta, tf.matmul(var[:, :, 27:, 27:], new_delta), transpose_a=True)
        # normal_cost = tf.reshape(normal_cost, shape=(S, N))
        

        return - 0.5 * dist_list #- self.log_normalization_constant(tf.squeeze(self.variance)) #* 500. - 0.5 * normal_cost * 10.

    @tf.function
    def _sample_config_cost(self, f):
        # Get the shape of the input tensor
        s, n, d = f.shape

        # Flatten the tensor so that it can be processed in one batch
        f = tf.reshape(f, (s * n, d))

        # Compute forward kinematics for all configurations in parallel using tf.vectorized_map
        k, ee = tf.vectorized_map(self._fk_cost, f)

        # tf.print(k.shape)
        # Reshape the output back to the original shape
        return tf.reshape(k, (s, n, self._p, 3)), tf.reshape(ee, (s, n, 4, 4))

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
