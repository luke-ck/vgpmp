import numpy as np


class RobotMixin(object):
    """
    Mixin class that serves as common base module that contains shared attributes
    and functionalities between the PyBullet and TensorFlow modules.
    """

    def __init__(self, robot_name, dh_parameters, dof, twist, fk_slice, craig_dh_convention, joint_limits, velocity_limits,
                 base_pose=None, *args, **kwargs):
        self.name = robot_name
        self.dof = dof
        self.transform_fn = None
        self.craig_notation = None
        self.DH = None
        self.twist = None
        self.velocity_limits = None
        self.joint_limits = None

        self.set_dh_parameters(dh_parameters, twist, craig_dh_convention)

        self.fk_slice = fk_slice
        self.set_joint_limits(joint_limits)
        self.set_velocity_limits(velocity_limits)
        self.base_pose = base_pose

        self.args = args
        # self.__dict__.update(kwargs)

    def forward_kinematics(self, thetas) -> np.array:
        base_homogenous_transform = self.base_pose
        assert self.base_pose is not None, "Base pose is not set"
        assert thetas.shape == (self.dof, 1), f"Expected shape: {(self.dof, 1)}, got {thetas.shape}"
        assert self.twist.shape == (self.dof, 1), f"Expected shape: {(self.dof, 1)}, got {self.twist.shape}"
        angles = thetas + self.twist
        dh_mat = np.hstack((angles, self.DH))

        transform_matrices = np.zeros((self.dof, 4, 4), dtype=np.float64)

        for idx, params in enumerate(dh_mat):
            transform_matrices[idx] = self.transform_fn(params[0], params[1], params[2], params[3])

        homogenous_transforms = np.zeros((len(thetas) + 1, 4, 4), dtype=np.float64)
        homogenous_transforms[0] = base_homogenous_transform

        for i in range(len(transform_matrices)):
            homogenous_transforms[i + 1] = homogenous_transforms[i] @ transform_matrices[i]

        homogenous_transforms = homogenous_transforms.reshape((-1, 4, 4))  # Reshape once

        return homogenous_transforms

    def get_transform_matrix(self, theta, d, a, alpha):
        """
            compute the homogenous transform matrix for a link given theta, d, a, alpha.
            Classic/Spong convention
            Args:
                theta (float): joint angle
                d (float): link length
                a (float): link offset from previous link
                alpha (float): link twist angle
            Returns:
                (array): homogenous transform matrix
        """
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        c_alpha = np.cos(alpha)
        s_alpha = np.sin(alpha)
        h = np.array([
            [c_theta, - s_theta * c_alpha, s_theta * s_alpha, a * c_theta],
            [s_theta, c_theta * c_alpha, - c_theta * s_alpha, a * s_theta],
            [0, s_alpha, c_alpha, d],
            [0, 0, 0, 1]
        ], dtype=np.float64)

        return h

    def get_transform_matrix_craig(self, theta, d, a, alpha):
        """
        Same as get_transform_matrix but with Modified/Craig convention
        """
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        c_alpha = np.cos(alpha)
        s_alpha = np.sin(alpha)

        h = np.array([
            [c_theta, -s_theta, 0, a],
            [s_theta * c_alpha, c_theta * c_alpha, -s_alpha, -d * s_alpha],
            [s_theta * s_alpha, c_theta * s_alpha, c_alpha, d * c_alpha],
            [0, 0, 0, 1]
        ], dtype=np.float64)

        return h

    def set_joint_limits(self, joint_limits):
        assert len(
            joint_limits) == 2 * self.dof, "Cannot set joint limits for a different number than the total active " \
                                           "joints"

        self.joint_limits = joint_limits

    def set_velocity_limits(self, param):
        assert len(param) == 2 * self.dof, "Velocity limits must be of length {}".format(self.dof)
        self.velocity_limits = param

    def set_dh_parameters(self, dh_parameters, twist, craig_dh_convention):
        self.DH = np.array(dh_parameters).reshape((-1, 3))
        self.twist = np.array(twist).reshape((-1, 1))
        self.craig_notation = craig_dh_convention
        self.transform_fn = self.get_transform_matrix_craig if self.craig_notation else self.get_transform_matrix
