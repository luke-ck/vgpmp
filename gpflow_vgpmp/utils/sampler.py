from gpflow import default_float
from gpflow_vgpmp.utils.parameter_loader import ParameterLoader
from gpflow_vgpmp.utils.robot import Robot, get_base
from gpflow_vgpmp.utils.robot_mixin import RobotMixin
import tensorflow as tf
import numpy as np

__all__ = "sampler"


def set_base(translation) -> np.array:
    assert len(translation) == 3
    rotation = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    T = get_base(rotation, translation)
    return T


class Sampler(RobotMixin):
    r"""
        This class is the interface that enables communication
        between tensorflow and pybullet. Cost for samples generated
        in the Monte Carlo routine is computed here, using custom
        gradients, to be able to call pybullet inside the computation
        graph.

    """

    def __init__(self, config: ParameterLoader, robot: Robot):
        robot_params = config.robot_params
        sphere_offsets = robot.sphere_offsets
        base_pose = robot.base_pose

        assert robot_params is not None
        assert sphere_offsets is not None and sphere_offsets is not []
        assert base_pose is not None and base_pose is not []

        super().__init__(**robot_params, vectorize=True)

        self.sphere_offsets = sphere_offsets  # if robot is set up properly this will be = as sphere_radii
        self.d = self.DH[:, 0].reshape(self.dof, 1)  # keep this in memory to avoid reshaping
        self.a = self.DH[:, 1].reshape(self.dof, 1)
        self.alpha = self.DH[:, 2].reshape(self.dof, 1)
        self.DH = tf.constant(self.DH, shape=(self.dof, 3), dtype=default_float())

        self.twist = tf.constant(self.twist, shape=(self.dof, 1), dtype=default_float())
        self.base_pose = tf.expand_dims(tf.constant(base_pose), axis=0)
        self.num_spheres_per_link = robot.num_spheres_per_link
        self.sphere_offsets = np.zeros((len(self.sphere_offsets), 4, 4))

        for index, offset in enumerate(sphere_offsets):
            mat = self.get_mat(self.name, index, offset)

            self.sphere_offsets[index] = mat

        self.sphere_offsets = tf.constant(self.sphere_offsets, shape=(len(sphere_offsets), 4, 4), dtype=default_float())
        self.joint_config_uncertainty = tf.ones(shape=(self.dof, 1), dtype=default_float())


    @tf.custom_gradient
    def check_gradients(self, x):
        def grad(upstream):
            upstream_string = tf.strings.format("{}\n", upstream, summarize=-1)
            tf.io.write_file("check_grads.txt", upstream_string)
            return upstream

        return x, grad

    def get_mat(self, robot_name, index, offset):
        if robot_name == "wam":
            if index < 8:
                return set_base((offset[0] - 0.045, -offset[1], offset[2]))
            elif 8 < index <= 12:
                return set_base((offset[0] + 0.045, -offset[1] - 0.05, offset[2]))
            elif index > 14:
                return set_base((offset[0], offset[1], offset[2]))
            elif index == 8:
                return set_base((0, 0, 0))
            else:
                return set_base((offset[0], -offset[1], offset[2]))
        elif robot_name == "ur10":
            if 0 < index < 7:
                return set_base((offset[2], offset[0], offset[1] + 0.163941 + 0.05))
            else:
                return set_base((offset[2], offset[0], offset[1]))
        elif robot_name == "kuka":
            if 1 < index < 5:
                return set_base((offset[0], -offset[2] + 0.18, offset[1]))
            elif 5 <= index < 8:
                return set_base((offset[0], offset[2], offset[1]))
            elif 8 <= index < 11:
                return set_base((offset[0], offset[2] - 0.18, -offset[1]))
            elif 11 <= index < 15:
                return set_base((offset[0], -offset[2], offset[1]))
            elif 15 <= index < 17:
                return set_base((offset[0], offset[2] + 0.1, offset[1] - 0.06))
            elif 17 <= index < 20:
                return set_base((offset[0], offset[2] - 0.07, offset[1]))
            else:
                return set_base((offset[0], offset[1], offset[2]))
        else:
            return set_base(offset)

    @tf.function(jit_compile=True)
    def forward_kinematics(self, thetas):
        # dh_mat = tf.concat([thetas + self.twist, self.DH], axis=-1)

        # Get the modified or standard transform matrix for each set of DH parameters
        # transform_matrices = tf.map_fn(
        #     lambda i: self.transform_fn(i[0], i[1], i[2], i[3]),
        #     dh_mat, fn_output_signature=default_float(), parallel_iterations=None)

        #
        transform_matrices = self.transform_fn(thetas + self.twist, self.d, self.a, self.alpha)

        homogeneous_transforms = tf.concat([self.base_pose, transform_matrices], axis=0)

        # Compute the matrix product of all the homogeneous transforms
        out = tf.scan(tf.matmul, homogeneous_transforms)

        return out

    @tf.function(jit_compile=True)
    def get_transform_matrix_scalar(self, theta, d, a, alpha):
        r"""
        Returns 4x4 homogenous matrix from DH parameters
        Arguments are expected to be scalars
        """
        c_theta = tf.cos(theta)
        s_theta = tf.sin(theta)
        c_alpha = tf.cos(alpha)
        s_alpha = tf.sin(alpha)

        h = tf.stack([
            [c_theta, -s_theta * c_alpha, s_theta * s_alpha, a * c_theta],
            [s_theta, c_theta * c_alpha, -c_theta * s_alpha, a * s_theta],
            [0., s_alpha, c_alpha, d],
            [0., 0., 0., 1.]
        ])

        return h

    @tf.function(jit_compile=True)
    def get_transform_matrix_vec(self, theta, d, a, alpha):
        r"""
        Returns dofx4x4 homogeneous matrix from DH parameters
        Args:
            theta: Tensor of shape (dof,) or (dof, 1)
            d: Tensor of shape (dof,) or (dof, 1)
            a: Tensor of shape (dof,) or (dof, 1)
            alpha: Tensor of shape (dof,) or (dof, 1)
        """
        # Make sure all input tensors have the same shape

        c_theta = tf.cos(theta)
        s_theta = tf.sin(theta)
        c_alpha = tf.cos(alpha)
        s_alpha = tf.sin(alpha)

        h = tf.stack([
            c_theta, -s_theta * c_alpha, s_theta * s_alpha, a * c_theta,
            s_theta, c_theta * c_alpha, -c_theta * s_alpha, a * s_theta,
            tf.zeros_like(theta), s_alpha, c_alpha, d,
            tf.zeros_like(theta), tf.zeros_like(theta), tf.zeros_like(theta), tf.ones_like(theta)
        ], axis=-1)

        h = tf.reshape(h, shape=[-1, 4, 4])

        return h

    @tf.function(jit_compile=True)
    def get_transform_matrix_craig_scalar(self, theta, d, a, alpha):
        r"""
        Returns 4x4 homogenous matrix from DH parameters using Craig convention
        Arguments are expected to be scalars
        """
        c_theta = tf.cos(theta)
        s_theta = tf.sin(theta)
        c_alpha = tf.cos(alpha)
        s_alpha = tf.sin(alpha)

        h = tf.stack([
            [c_theta, -s_theta, 0., a],
            [s_theta * c_alpha, c_theta * c_alpha, -s_alpha, -d * s_alpha],
            [s_theta * s_alpha, c_theta * s_alpha, c_alpha, d * c_alpha],
            [0., 0., 0., 1.]
        ])

        return h

    @tf.function(jit_compile=True)
    def get_transform_matrix_craig_vec(self, theta, d, a, alpha):
        r"""
        Returns dofx4x4 homogenous matrix from DH parameters using Craig convention
        Args:
            theta: Tensor of shape (dof,) or (dof, 1)
            d: Tensor of shape (dof,) or (dof, 1)
            a: Tensor of shape (dof,) or (dof, 1)
            alpha: Tensor of shape (dof,) or (dof, 1)
        """
        c_theta = tf.cos(theta)
        s_theta = tf.sin(theta)
        c_alpha = tf.cos(alpha)
        s_alpha = tf.sin(alpha)

        h = tf.stack([
            c_theta, -s_theta, tf.zeros_like(theta), a,
            s_theta * c_alpha, c_theta * c_alpha, -s_alpha, -d * s_alpha,
            s_theta * s_alpha, c_theta * s_alpha, c_alpha, d * c_alpha,
            tf.zeros_like(theta), tf.zeros_like(theta), tf.zeros_like(theta), tf.ones_like(theta)
        ], axis=-1)

        h = tf.reshape(h, shape=[-1, 4, 4])

        return h

    @tf.function(jit_compile=True)
    def forward_kinematics_cost(self, joint_config):
        r""" Computes the cost for a given config. It is advised to
        only use this function with the autograph since the decorator
        will change the joint configuration of the robot. For other
        practices (such as debugging) a numpy equivalent is available
        in the Robot class.

        Args:
            joint_config ([tf.tensor]): D x 1

        Returns:
            sphere_loc [tf.tensor]: P x 3
        """

        # <------------- Computing Forward Kinematics ------------>
        # joint_config = self.check_gradients(joint_config)
        fk_pos = self._forward_kinematics_joints_to_spheres(joint_config)
        sphere_positions = fk_pos @ self.sphere_offsets
        return tf.squeeze(sphere_positions[:, :3, 3])  # just return the positions

    @tf.function
    def _forward_kinematics_joints_to_spheres(self, joint_config):
        """
        Compute the forward kinematics from joint to sphere positions
        """
        fk_pos = tf.gather(self.forward_kinematics(joint_config), self.fk_slice, axis=0)
        fk_pos = tf.repeat(fk_pos, repeats=self.num_spheres_per_link, axis=0)
        return fk_pos

    @tf.function
    def compute_joint_pos_uncertainty(self, joint_config, joint_config_uncertainty):
        r"""Computes the uncertainties for a given joint config. The function
        needs the use of autograph.

        Args:
            joint_config ([tf.tensor]): D x 1
            joint_config_uncertainty ([tf.tensor]): D x 1
        returns:
            position_uncertainties [tf.tensor]: D x 3
        """
        joint_config = tf.reshape(joint_config, (7, 1))
        joint_config_uncertainty = tf.reshape(joint_config_uncertainty, (7, 1))

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(joint_config)
            fk_pos = self.forward_kinematics(joint_config)
            xyz_positions = fk_pos[-1, :3, 3]
            x = xyz_positions[0]
            y = xyz_positions[1]
            z = xyz_positions[2]
        gradients = tf.stack([tape.gradient(x, joint_config),
                              tape.gradient(y, joint_config),
                              tape.gradient(z, joint_config)])

        position_uncertainties = gradients * joint_config_uncertainty[None, ...]
        position_uncertainties = tf.squeeze(position_uncertainties ** 2)
        return tf.math.sqrt(tf.reduce_sum(position_uncertainties, axis=-1))

    @tf.function
    def get_idx(self, link, link_array):
        return tf.squeeze(tf.where(tf.equal(link, link_array)))
