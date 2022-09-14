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

    def __init__(self, robot, parameters):
        self.robot = robot
        link_offsets = self.robot.joint_link_offsets
        sphere_offsets = self.robot.sphere_offsets
        self.DH = tf.constant(parameters["dh_parameters"], shape=(self.robot.dof, 3), dtype=default_float())
        self.pi = tf.reshape(tf.constant(parameters["pi"], dtype=default_float()), (self.robot.dof, 1))
        self.arm_base = tf.expand_dims(tf.constant(self.robot.base_pose), axis=0)
        self.spheres_to_links = np.array(self.robot.sphere_link_interval)
        self.num_spheres = self.robot.num_spheres
        link_offsets = np.repeat(link_offsets, repeats=self.num_spheres, axis=0)
        link_sphere_offsets = link_offsets + sphere_offsets
        self.link_sphere_offsets = np.zeros((len(link_sphere_offsets), 4, 4))

        for index, offset in enumerate(link_sphere_offsets):
            mat = set_base(offset)
            self.link_sphere_offsets[index] = mat

        self.link_sphere_offsets = tf.convert_to_tensor(self.link_sphere_offsets, dtype=default_float())

    @tf.custom_gradient
    def check_gradients(self, input):
        def grad(upstream):
            one_string = tf.strings.format("{}\n", (upstream), summarize=-1)
            tf.io.write_file("matrices.txt", one_string)
            # tf.print("upstream translation dim:", upstream.shape)
            return upstream

        return input, grad

    @tf.function
    def _compute_fk(self, joint_config):
        DH = tf.concat([joint_config + self.pi, self.DH], axis=-1)

        transform_matrices = tf.map_fn(
            lambda i: get_transform_matrix(i[0], i[1], i[2], i[3]), DH, fn_output_signature=default_float(),
            parallel_iterations=4)

        homogeneous_transforms = tf.concat(
            [self.arm_base, transform_matrices], axis=0)

        out = tf.scan(tf.matmul, homogeneous_transforms)

        return out[1:]  # don't return the static base

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

        fk_pos = self._compute_fk(joint_config)
        # tf.print(fk_pos)
        fk_pos = tf.repeat(fk_pos, repeats=self.num_spheres, axis=0)
        fk_pos = tf.matmul(self.link_sphere_offsets, fk_pos)

        # fk_pos = tf.matmul(fk_pos, self.link_sphere_offsets)
        # tf.print(self.link_sphere_offsets, summarize=-1)
        # tf.print(self.num_spheres)
        # out = tf.math.add(out, self.joint_to_link_offsets)
        #
        # J = jacobian(spheres_loc, out)
        # rows = tf.range(out.shape[0])
        # cols = tf.range(out.shape[1])
        #
        # ii = tf.range(out.shape[0])
        # to_check_plus = tf.map_fn(lambda i: self.per_row_plus(out, i), ii, fn_output_signature=default_float())
        # to_check_minus = tf.map_fn(lambda i: self.per_row_minus(out, i), ii, fn_output_signature=default_float())

        # func1 = tf.range(to_check_plus.shape[0])
        # func2 = tf.range(to_check_plus.shape[1])
        #
        # up_shift = tf.map_fn(lambda i: tf.map_fn(lambda j: self.joint_to_spheres(to_check_plus[i][j]), func2,
        #                                     fn_output_signature=default_float()),
        #                 func1, fn_output_signature=default_float())
        #
        # down_shift = tf.map_fn(lambda i: tf.map_fn(lambda j: self.joint_to_spheres(to_check_minus[i][j]), func2,
        #                                               fn_output_signature=default_float()),
        #                           func1, fn_output_signature=default_float())
        #
        # finite_diff = (up_shift - down_shift) / (2 * 1e-5)
        #
        # one_string = tf.strings.format("{}\n FINITE DIFF is HERE \n{}\n", (J, finite_diff), summarize=-1)
        # tf.io.write_file("fk_gradients.txt", one_string)

        return tf.squeeze(fk_pos[:, :3, 3])

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

    @tf.function
    def sampleConfigs(self, joint_configs, sample_dim):
        r"""

        Args:
            joint_configs ([tf.tensor]): S x D
            sample_dim ([type]): S

        Returns:
            [type]: S x P x 3
        """

        K = tf.map_fn(lambda i: self._fk_cost(tf.reshape(
            joint_configs[i], (-1, 1))), tf.range(sample_dim), fn_output_signature=default_float(),
                      parallel_iterations=None)

        return K
