import queue
import threading
from unittest.mock import MagicMock

import pytest
import pybullet as p
import numpy as np
from data.problemsets.problemset import create_problems
from gpflow_vgpmp.utils.robot import Robot
from gpflow_vgpmp.utils.sampler import Sampler
from gpflow_vgpmp.utils.simulation import ParameterLoader, Simulation, SimulationThread
from gpflow_vgpmp.utils.simulator import RobotSimulator


@pytest.fixture(scope="session")
def pybullet_session_fixture():
    env = RobotSimulator(parameter_file_path='./test_params.yaml')

    robot = env.robot

    params = env.get_simulation_params()
    robot_params = params.robot_params
    print(robot_params['robot_name'])
    queries, planner_params, joint_names, default_pose, default_robot_pos_and_orn = create_problems(
        problemset_name=params.scene_params["problemset"], robot_name=robot_params['robot_name'])

    default_pose = np.array([0] * robot_params['dof']).reshape(1, robot_params['dof'])
    start_config = np.array([0] * robot_params['dof']).reshape(1, robot_params['dof'])
    robot.initialise(default_robot_pos_and_orn=default_robot_pos_and_orn,
                     start_config=start_config,
                     joint_names=joint_names,
                     default_pose=default_pose,
                     benchmark=False)

    return env, planner_params


@pytest.fixture
def mock_input_config():
    return [
        {
            'robot': {
                'robot_name': 'wam'
            }
        },
        {
            "scene": {
                "position": [-0.2, 0.0, 0.08],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "environment_name": "industrial",
                "objects": [],
                "objects_position": [],
                "objects_orientation": [],
                "problemset": "industrial",
                "sdf_name": "industrial_vgpmp"
            }
        },
        {
            "trainable_parameters": {
                "q_mu": True,
                "q_sqrt": True,
                "lengthscales": True,
                "kernel_variance": True,
                "sigma_obs": False,
                "inducing_variable": False,
                "alpha": False
            }
        },
        {
            "graphics": {
                "visuals": False,
                "debug_joint_positions": False,
                "debug_sphere_positions": False,
                "visualize_best_sample": True,
                "visualize_ee_path_uncertainty": False,
                "visualize_drawn_samples": False
            }
        }
    ]


@pytest.fixture
def mock_final_config():
    # Create a mock configuration dictionary
    return {
        "robot_params": {
            'radius': [0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.1, 0.0675, 0.0675, 0.0675, 0.0675,
                       0.0675, 0.0675, 0.0675, 0.034, 0.034, 0.034, 0.034, 0.034, 0.034, 0.034, 0.034, 0.034],
            'num_spheres': 25,
            'active_links': ['wam/base_link', 'wam/shoulder_yaw_link', 'wam/shoulder_pitch_link', 'wam/upper_arm_link',
                             'wam/forearm_link', 'wam/wrist_yaw_link', 'wam/wrist_pitch_link'],
            'active_joints': ['wam/base_yaw_joint', 'wam/shoulder_pitch_joint', 'wam/shoulder_yaw_joint',
                              'wam/elbow_pitch_joint', 'wam/wrist_yaw_joint', 'wam/wrist_pitch_joint',
                              'wam/palm_yaw_joint'],
            'link_name_base': 'wam/base_link',
            'link_name_wrist': 'wam/wrist_pitch_link',
            'path': 'wam.urdf',
            'joint_limits': [2.6, -2.6, 2.0, -2.0, 2.8, -2.8, 3.1, -0.9, 1.24, -4.76, 1.6, -1.6, 3.0, -3.0],
            'velocity_limits': [2.6, -2.6, 2.0, -2.0, 2.8, -2.8, 3.1, -0.9, 1.24, -4.76, 1.6, -1.6, 3.0, -3.0],
            'dh_parameters': [0.0, 0.0, -1.5708, 0.0, 0.0, 1.5708, 0.55, 0.045, -1.5708, 0.0, -0.045, 1.5708, 0.3, 0.0,
                              -1.5708, 0.0, 0.0, 1.5708, 0.06, 0.0, 0.0],
            'twist': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'dof': 7,
            'craig_dh_convention': False,
            'no_frames_for_spheres': 4,
            'fk_slice': [3, 5, 6, 7],
            'q_mu': 'None',
            'urdf_path': '/home/lucasc/git/vgpmp/data//robots/wam/wam.urdf',
            'robot_name': 'wam'
        },
        'planner_params': {
            'sigma_obs': 0.005,
            'epsilon': 0.05,
            'lengthscales': [5.0, 5.0, 5.0, 2.0, 5.0, 5.0, 5.0],
            'variance': 0.25,
            'alpha': 100,
            'num_samples': 7,
            'num_inducing': 24,
            'learning_rate': 0.09,
            'num_steps': 130,
            'time_spacing_X': 70,
            'time_spacing_Xnew': 150
        },
        "scene_params": {
            'position': [-0.2, 0.0, 0.08],
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'environment_name': 'industrial',
            'objects': [],
            'objects_position': [],
            'objects_orientation': [],
            'problemset': 'industrial',
            'sdf_name': 'industrial_vgpmp',
            'object_path': [],
            'sdf_path': '/home/lucasc/git/vgpmp/data//scenes/industrial/industrial_vgpmp.sdf',
            'environment_path': '/home/lucasc/git/vgpmp/data//scenes/industrial/industrial.urdf'
        },
        "trainable_parameters": {
            'q_mu': True,
            'q_sqrt': True,
            'lengthscales': True,
            'kernel_variance': True,
            'sigma_obs': False,
            'inducing_variable': False,
            'alpha': False
        },
        "graphics": {
            'visuals': False,
            'debug_joint_positions': False,
            'debug_sphere_positions': False,
            'visualize_best_sample': True,
            'visualize_ee_path_uncertainty': False,
            'visualize_drawn_samples': False
        }
    }


@pytest.fixture
def mock_parameter_loader(mock_input_config, mock_final_config):
    # Create a mock parameter loader with the given configuration
    parameter_loader = ParameterLoader('./test_params.yaml')
    parameter_loader.params = mock_input_config
    parameter_loader.set_params = MagicMock(return_value=mock_final_config)
    return parameter_loader


@pytest.fixture
def mock_simulation_thread(mock_input_config):
    graphic_params = mock_input_config[-1]['graphics']

    # Create a mock simulation thread
    simulation_thread = SimulationThread(graphic_params, thread_ready_event=threading.Event(), queue=queue.Queue())
    simulation_thread.start = MagicMock()
    simulation_thread.stop = MagicMock()
    simulation_thread.join = MagicMock()
    simulation_thread.is_alive = MagicMock(return_value=True)
    return simulation_thread


@pytest.fixture
def mock_simulation(mock_input_config, mock_simulation_thread):
    graphic_params = mock_input_config[-1]['graphics']
    # Create a mock simulation with the mock simulation thread
    simulation = Simulation(graphic_params)

    # Modify the start_simulation_thread method to invoke the run method of mock_simulation_thread
    def start_simulation_thread_mock():
        mock_simulation_thread.run()

    simulation.start_simulation_thread = MagicMock(side_effect=start_simulation_thread_mock)
    simulation.stop_simulation_thread = MagicMock()
    simulation.check_events = MagicMock()
    simulation.check_simulation_thread_health = MagicMock(return_value=True)
    return simulation