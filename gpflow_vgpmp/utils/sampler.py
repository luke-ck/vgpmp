from gpflow_vgpmp.utils.ops import *


@tf.function
def get_transform_matrix(theta, d, a, alpha):
    r"""
    Returns 4x4 homogenous matrix from DH parameters
    """
    cTheta = tf.math.cos(theta)
    sTheta = tf.math.sin(theta)
    calpha = tf.math.cos(alpha)
    salpha = tf.math.sin(alpha)

    T = tf.stack([
        tf.cast([cTheta, -sTheta * calpha, sTheta *
                 salpha, a * cTheta], dtype=default_float()),
        tf.cast([sTheta, cTheta * calpha, -cTheta *
                 salpha, a * sTheta], dtype=default_float()),
        tf.cast([0., salpha, calpha, d], dtype=default_float()),
        tf.cast([0., 0., 0., 1.], dtype=default_float())
    ])

    return T


@tf.function
def get_modified_transform_matrix(theta, d, a, alpha):
    r"""
    Returns 4x4 homogenous matrix from DH parameters
    """
    cTheta = tf.math.cos(theta)
    sTheta = tf.math.sin(theta)
    calpha = tf.math.cos(alpha)
    salpha = tf.math.sin(alpha)

    T = tf.stack([
        tf.cast([cTheta, -sTheta, 0., a], dtype=default_float()),
        tf.cast([sTheta * calpha, cTheta * calpha, -salpha, -d * salpha], dtype=default_float()),
        tf.cast([sTheta * salpha, cTheta * salpha, calpha, d * calpha], dtype=default_float()),
        tf.cast([0., 0., 0., 1.], dtype=default_float())
    ])

    return T


def translation_vector(position):
    return np.concatenate([position, [1]]).reshape((4, 1))


__all__ = "sampler"


class Sampler:
    r"""
        This class is the interface that enables communication
        between tensorflow and pybullet. Cost for samples generated
        in the Monte Carlo routine is computed here, using custom
        gradients, to be able to call pybullet inside the computation
        graph.

    """

    def __init__(self, robot, parameters, robot_name):
        self.robot = robot
        sphere_offsets = robot.sphere_offsets
        self.DH = tf.constant(parameters["dh_parameters"], shape=(self.robot.dof, 3), dtype=default_float())
        self.pi = tf.reshape(tf.constant(parameters["twist"], dtype=default_float()), (self.robot.dof, 1))
        self.arm_base = tf.expand_dims(tf.constant(self.robot.base_pose), axis=0)
        self.spheres_to_links = np.array(self.robot.sphere_link_interval)
        self.num_spheres = robot.num_spheres
        self.craig_dh_convention = parameters["craig_dh_convention"]
        self.sphere_offsets = np.zeros((len(sphere_offsets), 4, 4))
        self.fk_slice = parameters["fk_slice"]

        for index, offset in enumerate(sphere_offsets):
            if robot_name == "wam":
                if index < 8:
                    mat = set_base((offset[0] - 0.045, -offset[1], offset[2]))
                elif index > 8 and index <= 12:
                    mat = set_base((offset[0] + 0.045, -offset[1] - 0.05, offset[2]))
                elif index > 14:
                    mat = set_base((offset[0], offset[1], offset[2]))
                elif index == 8:
                    mat = set_base((0, 0, 0))
                else:
                    mat = set_base((offset[0], -offset[1], offset[2]))

            elif robot_name == "ur10":
                if index > 0 and index < 7:
                    mat = set_base((offset[2], offset[0], offset[1] + 0.163941 + 0.05))
                else:
                    mat = set_base((offset[2], offset[0], offset[1]))

            elif robot_name == "kuka":
                if index > 1 and index < 5:
                    mat = set_base((offset[0], -offset[2] + 0.18, offset[1]))
                elif index >= 5 and index < 8:
                    mat = set_base((offset[0], offset[2], offset[1]))
                elif index >= 8 and index < 11:
                    mat = set_base((offset[0], offset[2] - 0.18, -offset[1]))
                elif index >= 11 and index < 15:
                    mat = set_base((offset[0], -offset[2], offset[1]))
                elif index >= 15 and index < 17:
                    mat = set_base((offset[0], offset[2] + 0.1, offset[1] - 0.06))
                elif index >= 17 and index < 20:
                    mat = set_base((offset[0], offset[2] - 0.07, offset[1]))
                else:
                    mat = set_base((offset[0], offset[1], offset[2]))
            else:
                mat = set_base(offset)
            self.sphere_offsets[index] = mat

        self.sphere_offsets = tf.constant(self.sphere_offsets, shape=(len(sphere_offsets), 4, 4), dtype=default_float())
        self.joint_config_uncertainty = tf.constant([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], shape=(7, 1),
                                                    dtype=default_float()) * 10

    @tf.custom_gradient
    def check_gradients(self, x):
        def grad(upstream):
            upstream_string = tf.strings.format("{}\n", upstream, summarize=-1)
            tf.io.write_file("matrices.txt", upstream_string)
            return upstream

        return x, grad

    @tf.function
    def _compute_fk(self, joint_config):
        DH = tf.concat([joint_config + self.pi, self.DH], axis=-1)

        if self.craig_dh_convention:
            transform_matrices = tf.map_fn(
                lambda i: get_modified_transform_matrix(i[0], i[1], i[2], i[3]), DH, fn_output_signature=default_float(),
                parallel_iterations=4)
        else:
            transform_matrices = tf.map_fn(
                lambda i: get_transform_matrix(i[0], i[1], i[2], i[3]), DH, fn_output_signature=default_float(),
                parallel_iterations=4)

        homogeneous_transforms = tf.concat(
            [self.arm_base, transform_matrices], axis=0)

        out = tf.scan(tf.matmul, homogeneous_transforms)

        return out

    @tf.function
    def _compute_fk_ee_pos(self, joint_config):
        DH = tf.concat([joint_config + self.pi, self.DH], axis=-1)

        if self.craig_dh_convention:
            transform_matrices = tf.map_fn(
                lambda i: get_modified_transform_matrix(i[0], i[1], i[2], i[3]), DH, fn_output_signature=default_float(),
                parallel_iterations=4)
        else:
            transform_matrices = tf.map_fn(
                lambda i: get_transform_matrix(i[0], i[1], i[2], i[3]), DH, fn_output_signature=default_float(),
                parallel_iterations=4)

        homogeneous_transforms = tf.concat(
            [self.arm_base, transform_matrices], axis=0)

        out = tf.scan(tf.matmul, homogeneous_transforms)

        return out[-1, :3, 3]

    @tf.custom_gradient
    def check_gradients(self, x):

        def grad(upstream):
            tf.print(upstream, summarize=-1)
            return upstream

        return x, grad

    @tf.function
    def _fk_cost(self, joint_config):
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
        fk_pos = tf.gather(self._compute_fk(joint_config), self.fk_slice, axis=0)
        fk_pos = tf.repeat(fk_pos, repeats=self.num_spheres, axis=0)
        sphere_positions = fk_pos @ self.sphere_offsets  # hardcoded for now
        return tf.squeeze(sphere_positions[:, :3, 3])

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
            fk_pos = self._compute_fk(joint_config)
            xyz_positions = fk_pos[-1, :3, 3]
            func1 = tf.range(xyz_positions.shape[0])
            x = xyz_positions[0]
            y = xyz_positions[1]
            z = xyz_positions[2]
        gradients = tf.stack([tape.gradient(x, joint_config),
                              tape.gradient(y, joint_config),
                              tape.gradient(z, joint_config)])

        position_uncertainties = gradients * joint_config_uncertainty[None, ...]
        position_uncertainties = tf.squeeze(position_uncertainties ** 2)
        return tf.math.sqrt(tf.reduce_sum(position_uncertainties, axis=-1))

    @tf.custom_gradient
    def joint_to_spheres(self, joints):
        spheres_loc = tf.py_function(self.robot.get_sphere_transform, [joints], default_float())

        def grad(upstream):
            func = tf.range(self.spheres_to_links.shape[0])
            gr = tf.map_fn(
                lambda i: tf.reduce_sum(
                    upstream[
                    tf.gather_nd(self.spheres_to_links, [i, 0]): tf.gather_nd(self.spheres_to_links, [i, 1])
                    ], axis=0
                ), func, fn_output_signature=default_float()
            )

            return gr

        return tf.reshape(spheres_loc, (-1, 3)), grad

    @tf.custom_gradient
    def joint_to_link(self, joints):
        r"""Maps joint positions to link positions
        Q - number of links for robot arm

        Args:
            joints ([tf.tensor]): D x 3

        Returns:
            links [tf.tensor]: Q x 3, grad [Tensor("gradients/...")]: D x 3
        """
        links = tf.py_function(self.robot._get_link_world_pos, [
            self.robot.sphere_idx], default_float())

        # tf.print(links)

        def get_idx_np(idx, arr):
            def f(x): return np.squeeze(np.where(np.equal(idx[x], arr)))

            return np.array([f(i) for i in range(len(idx))])

        def grad(upstream):
            return tf.stack(
                [
                    tf.reduce_sum(
                        tf.gather(upstream, get_idx_np(v, self.robot.sphere_idx)) *
                        tf.gather(joints, get_idx_np(
                            [k], self.robot.joint_idx)),
                        axis=0
                    )
                    for k, v in self.robot.joint_to_sampling_links.items()
                ]
            )

        return tf.reshape(links, (-1, 3)), grad

    @tf.function
    def get_idx(self, link, link_array):
        return tf.squeeze(tf.where(tf.equal(link, link_array)))

    @tf.custom_gradient
    def link_fk_to_spheres(self, links):
        r"""Computes the sphere location for the links found previously

        Args:
            links ([tf.tensor]): Q x 3

        Returns:
            spheres_loc [tf.tensor]: P x 3, grad [Tensor("gradients/...")]: Q x 3
        """
        spheres_loc = tf.py_function(self.robot.get_sphere_transform, [tf.py_function(self.robot._get_link_world_pos, [
            self.robot.sphere_idx], default_float())], default_float())

        def grad(upstream):
            func = tf.range(self.spheres_to_links.shape[0])
            gr = tf.map_fn(
                lambda i: tf.reduce_sum(
                    upstream[
                    tf.gather_nd(self.spheres_to_links, [i, 0]): tf.gather_nd(self.spheres_to_links, [i, 1])
                    ], axis=0
                ), func, fn_output_signature=default_float(), parallel_iterations=8
            )

            return gr

        return tf.reshape(spheres_loc, (-1, 3)), grad
