import numpy as np
import pytest

from gpflow_vgpmp.utils.parameter_loader import ParameterLoader
import pybullet as p


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


def test_np_tensorflow_fk(mock_sampler_with_robot):
    mock_initialized_robot, mock_sampler = mock_sampler_with_robot
    joint_config = np.array([0.1] * mock_initialized_robot.dof, dtype=np.float64).reshape(-1, 1)

    np_joint_mat = mock_initialized_robot.forward_kinematics(joint_config)

    tf_joint_mat = mock_sampler.forward_kinematics(joint_config).numpy()

    assert np.allclose(np_joint_mat, tf_joint_mat)
