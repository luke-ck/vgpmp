from unittest.mock import MagicMock
import numpy as np
import tensorflow as tf


def test_robot_mixin(mock_robot):

    # assert that transform_fn is set correctly by the craig_notation flag
    robot, mock_simulation = mock_robot
    parameter_loader, simulation = mock_simulation
    assert robot.craig_notation == parameter_loader.robot_params['craig_dh_convention']

    # this is the expected homogenous transform matrices if the classic DH convention is used
    h01 = np.array([[1.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
                    [0.0000000e+00, -3.6732051e-06, -1.0000000e+00, 0.0000000e+00],
                    [0.0000000e+00, 1.0000000e+00, -3.6732051e-06, 1.2730000e-01],
                    [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]], dtype=np.float64)

    h12 = np.array([[1.0000000e+00, -0.0000000e+00, 0.0000000e+00, -6.1200000e-01],
                    [0.0000000e+00, 1.0000000e+00, -0.0000000e+00, -0.0000000e+00],
                    [0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 0.0000000e+00],
                    [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]], dtype=np.float64)

    h23 = np.array([[1.0000000e+00, -0.0000000e+00, 0.0000000e+00, -5.7230000e-01],
                    [0.0000000e+00, 1.0000000e+00, -0.0000000e+00, -0.0000000e+00],
                    [0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 0.0000000e+00],
                    [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]], dtype=np.float64)

    h34 = np.array([[1.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
                    [0.0000000e+00, -3.6732051e-06, -1.0000000e+00, 0.0000000e+00],
                    [0.0000000e+00, 1.0000000e+00, -3.6732051e-06, 1.6394100e-01],
                    [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]], dtype=np.float64)

    h45 = np.array([[1.0000000e+00, 0.0000000e+00, -0.0000000e+00, 0.0000000e+00],
                    [0.0000000e+00, -3.6732051e-06, 1.0000000e+00, 0.0000000e+00],
                    [0.0000000e+00, -1.0000000e+00, -3.6732051e-06, 1.1570000e-01],
                    [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]], dtype=np.float64)

    h56 = np.array([[1.0000000e+00, -0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
                    [0.0000000e+00, 1.0000000e+00, -0.0000000e+00, 0.0000000e+00],
                    [0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 9.2200000e-02],
                    [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]], dtype=np.float64)

    robot.transform_fn = MagicMock(side_effect=[h01, h12, h23, h34, h45, h56])

    joint_config = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((-1, 1))

    robot.forward_kinematics(joint_config)

    assert robot.transform_fn.call_count == robot.dof

    simulation.stop_simulation_thread()


def test_initialize_robot(mock_robot):
    robot, mock_simulation = mock_robot
    parameter_loader, simulation = mock_simulation

    robot_pos_and_orn = parameter_loader.scene_params['robot_pos_and_orn']
    joint_names = parameter_loader.robot_params['joint_names']
    benchmark = parameter_loader.scene_params['benchmark']
    ur10_joint_link_offsets = np.array([[0., 0., 0., ],
                                        [0., 0., 0.306],
                                        [0., 0., 0.28615],
                                        [0., 0., 0., ],
                                        [0., 0., 0., ],
                                        [0., 0., 0., ]])

    if benchmark:
        base_pose = np.array([[-1., 0., 0., 0.],
                              [0., -1., 0., 0.],
                              [0., 0., 1., 0.],
                              [0., 0., 0., 1.]])

    else:
        base_pose = np.array([[1., 0., 0., 0.],
                              [0., 1., 0., 0.],
                              [0., 0., 1., 0.],
                              [0., 0., 0., 1.]])
    # Assertions
    assert robot.base_index == 0
    assert robot.active_joint_names == joint_names

    assert np.alltrue(robot.joint_link_offsets == ur10_joint_link_offsets)
    assert np.alltrue(robot.get_base_pose() == base_pose)
    assert robot.position == robot_pos_and_orn[0]
    assert robot.orientation == robot_pos_and_orn[1]
    assert robot.is_initialized
    assert robot.joint_limits == parameter_loader.robot_params['joint_limits']
    assert robot.velocity_limits == parameter_loader.robot_params['velocity_limits']

    simulation.stop_simulation_thread()


def test_np_tensorflow_fk(mock_sampler):
    robot, sampler, mock_simulation = mock_sampler
    parameter_loader, simulation = mock_simulation

    joint_config = np.array([0.1] * robot.dof, dtype=np.float64).reshape(-1, 1)

    np_joint_mat = robot.forward_kinematics(joint_config)

    tf_joint_mat = sampler.forward_kinematics(joint_config).numpy()

    assert np.allclose(np_joint_mat, tf_joint_mat)

    simulation.stop_simulation_thread()


def test_transform_matrix_np_tensorflow(mock_sampler):
    robot, sampler, mock_simulation = mock_sampler
    parameter_loader, simulation = mock_simulation
    assert robot.dof == sampler.dof
    assert np.alltrue(robot.DH == sampler.DH)
    assert np.alltrue(robot.twist == sampler.twist)
    assert np.alltrue(robot.base_pose == sampler.base_pose)

    joint_config = np.array([0.0] * robot.dof, dtype=np.float64).reshape(-1, 1)
    angles = joint_config + robot.twist
    np_dh_mat = np.hstack((angles, robot.DH))

    np_transform_matrices = np.zeros((robot.dof, 4, 4), dtype=np.float64)

    for idx, params in enumerate(np_dh_mat):
        np_transform_matrices[idx] = robot.transform_fn(params[0], params[1], params[2], params[3])

    tf_dh_mat = tf.constant(np_dh_mat, dtype=tf.float64)

    tf_transform_matrices = tf.map_fn(
        lambda i: sampler.transform_fn(i[0], i[1], i[2], i[3]),
        tf_dh_mat, fn_output_signature=tf.float64, parallel_iterations=None)

    assert np.allclose(np_transform_matrices, tf_transform_matrices.numpy())

    simulation.stop_simulation_thread()
