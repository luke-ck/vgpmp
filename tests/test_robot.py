import numpy as np
import tensorflow as tf

def test_initialize_robot(mock_robot_and_parameter_loader):
    mock_robot, mock_parameter_loader = mock_robot_and_parameter_loader

    robot_pos_and_orn = mock_parameter_loader.scene_params['robot_pos_and_orn']
    joint_names = mock_parameter_loader.robot_params['joint_names']
    default_pose = mock_parameter_loader.robot_params['default_pose']
    benchmark = mock_parameter_loader.scene_params['benchmark']
    ur10_joint_link_offsets = np.array([[0., 0., 0., ],
                                        [0., 0., 0.306],
                                        [0., 0., 0.28615],
                                        [0., 0., 0., ],
                                        [0., 0., 0., ],
                                        [0., 0., 0., ]])

    mock_robot.initialise(default_robot_pos_and_orn=robot_pos_and_orn,
                          joint_names=joint_names,
                          default_pose=default_pose,
                          benchmark=benchmark)

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
    assert mock_robot.base_index == 0
    assert mock_robot.active_joint_names == joint_names

    assert np.alltrue(mock_robot.joint_link_offsets == ur10_joint_link_offsets)
    assert np.alltrue(mock_robot.get_base_pose() == base_pose)


def test_np_tensorflow_fk(mock_sampler_with_robot_with_parameters):
    mock_initialized_robot, mock_sampler = mock_sampler_with_robot_with_parameters
    joint_config = np.array([0.1] * mock_initialized_robot.dof, dtype=np.float64).reshape(-1, 1)

    np_joint_mat = mock_initialized_robot.forward_kinematics(joint_config)

    tf_joint_mat = mock_sampler.forward_kinematics(joint_config).numpy()

    assert np.allclose(np_joint_mat, tf_joint_mat)


def test_transform_matrix_np_tensorflow(mock_sampler_with_robot_with_parameters):
    mock_initialized_robot, mock_sampler = mock_sampler_with_robot_with_parameters

    assert mock_initialized_robot.dof == mock_sampler.dof
    assert np.alltrue(mock_initialized_robot.DH == mock_sampler.DH)
    assert np.alltrue(mock_initialized_robot.twist == mock_sampler.twist)
    assert np.alltrue(mock_initialized_robot.base_pose == mock_sampler.base_pose)

    joint_config = np.array([0.1] * mock_initialized_robot.dof, dtype=np.float64).reshape(-1, 1)
    angles = joint_config + mock_initialized_robot.twist
    np_dh_mat = np.hstack((angles, mock_initialized_robot.DH))

    np_transform_matrices = np.zeros((mock_initialized_robot.dof, 4, 4), dtype=np.float64)

    for idx, params in enumerate(np_dh_mat):
        np_transform_matrices[idx] = mock_initialized_robot.transform_fn(params[0], params[1], params[2], params[3])

    tf_dh_mat = tf.constant(np_dh_mat, dtype=tf.float64)

    tf_transform_matrices = tf.map_fn(
        lambda i: mock_sampler.transform_fn(i[0], i[1], i[2], i[3]),
        tf_dh_mat, fn_output_signature=tf.float64, parallel_iterations=None)

    assert np.allclose(np_transform_matrices, tf_transform_matrices.numpy())
