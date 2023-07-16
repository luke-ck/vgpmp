import threading
import time
from pathlib import Path
from unittest.mock import MagicMock
import pytest
import numpy as np

from gpflow_vgpmp.utils.miscellaneous import get_root_package_path
from gpflow_vgpmp.utils.robot import Robot
from gpflow_vgpmp.utils.sampler import Sampler
from gpflow_vgpmp.utils.sdf_utils import SignedDistanceField
from gpflow_vgpmp.utils.simulation import Simulation, SimulationThread
from gpflow_vgpmp.utils.scene import Scene
from gpflow_vgpmp.utils.parameter_loader import ParameterLoader
from gpflow_vgpmp.utils.simulation_manager import SimulationManager


def initialized_simulation(parameter_loader):
    simulation = Simulation(parameter_loader)
    simulation.initialize()
    return simulation


def initialized_parameter_loader(mock_input_config):
    parameter_loader = ParameterLoader()
    parameter_loader.initialize(params=mock_input_config)
    return parameter_loader


@pytest.fixture
def test_params_path():
    return Path(get_root_package_path()) / 'tests' / 'test_params.yaml'


@pytest.fixture(params=[
    [
        {
            'robot': {
                'robot_name': "ur10"
            }
        },
        {
            "scene": {
                "position": [-0.2, 0.0, 0.08],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "environment_name": "industrial",
                "environment_file_name": "industrial",
                "sdf_file_name": "industrial_vgpmp",
                "objects": [],
                "objects_position": [],
                "objects_orientation": [],
                "benchmark": benchmark,
                "non_benchmark_attributes": {
                    "states": [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    ],
                    "robot_pos_and_orn": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                    "planner_params":
                        {
                            "sigma_obs": 0.005,
                            "epsilon": 0.05,
                            "lengthscales": [5.0, 5.0, 5.0, 2.0, 5.0, 5.0],
                            "variance": 0.25,
                            "alpha": 100,
                            "num_samples": 7,
                            "num_inducing": 24,
                            "learning_rate": 0.09,
                            "num_steps": 130,
                            "time_spacing_X": 70,
                            "time_spacing_Xnew": 150
                        }
                },
                "benchmark_attributes": {
                    "problemset_name": "testing"
                }
            }
        },
        {
            "trainable_params": {
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
    for benchmark in [True, False]
])
def mock_input_config(request):
    return request.param


@pytest.fixture(params=[
    {
        'robot_params': {
            "radius": [0.15, 0.13, 0.085, 0.085, 0.085, 0.085, 0.13, 0.1, 0.07, 0.07, 0.07, 0.07, 0.07, 0.1, 0.08, 0.08,
                       0.05],
            'num_spheres': 17,
            'joint_names': ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint',
                            'wrist_2_joint', 'wrist_3_joint'],
            'default_pose': [0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
            'active_links': ['shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link',
                             'wrist_3_link'],
            'active_joints': ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint',
                              'wrist_2_joint', 'wrist_3_joint'],
            'link_name_base': 'base_link', 'link_name_wrist': 'ee_link', 'path': 'ur10.urdf',
            'joint_limits': [6.28, -6.28, 6.28, -6.28, 6.28, -6.28, 6.28, -6.28, 6.28, -6.28, 6.28, -6.28],
            'velocity_limits': [6.28, -6.28, 6.28, -6.28, 6.28, -6.28, 6.28, -6.28, 6.28, -6.28, 6.28, -6.28],
            'dh_parameters': [0.1273, 0.0, 1.5708, 0.0, -0.612, 0.0, 0.0, -0.5723, 0.0, 0.163941, 0.0, 1.5708, 0.1157,
                              0.0, -1.5708, 0.0922, 0.0, 0.0],
            'twist': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'dof': 6,
            'craig_dh_convention': False,
            'num_frames_for_spheres': 5,
            'fk_slice': [1, 2, 3, 4, 5],
            'q_mu': [-1.507, -1.507, 0, 0, 0, 0],
            'robot_name': 'ur10'
        },
        'scene_params': {
            'position': [-0.2, 0.0, 0.08],
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'environment_name': 'industrial',
            'environment_file_name': 'industrial',
            'sdf_file_name': 'industrial_vgpmp',
            'objects': [],
            'objects_position': [],
            'objects_orientation': [],
            'benchmark': benchmark,
            'objects_path': [],
            'queries': queries,
            'robot_pos_and_orn': robot_pos_and_orn,
        },
        'planner_params': planner_params,
        'trainable_params': {
            'q_mu': True,
            'q_sqrt': True,
            'lengthscales': True,
            'kernel_variance': True,
            'sigma_obs': False,
            'inducing_variable': False,
            'alpha': False
        },
        'graphics_params': {
            'visuals': False,
            'debug_joint_positions': False,
            'debug_sphere_positions': False,
            'visualize_best_sample': True,
            'visualize_ee_path_uncertainty': False,
            'visualize_drawn_samples': False
        }
    }
    for benchmark, planner_params, queries, robot_pos_and_orn in (
            (
                    True, {
                        'sigma_obs': 0,
                        'epsilon': 0,
                        'lengthscales': [0] * 6,
                        'variance': 0,
                        'alpha': 0,
                        'num_samples': 0,
                        'num_inducing': 0,
                        'learning_rate': 0,
                        'num_steps': 0,
                        'time_spacing_X': 0,
                        'time_spacing_Xnew': 0
                    },
                    [([0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])],
                    ([0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0])
            ),
            (
                    False, {
                        'sigma_obs': 0.005,
                        'epsilon': 0.05,
                        'lengthscales': [5.0, 5.0, 5.0, 2.0, 5.0, 5.0],
                        'variance': 0.25,
                        'alpha': 100,
                        'num_samples': 7,
                        'num_inducing': 24,
                        'learning_rate': 0.09,
                        'num_steps': 130,
                        'time_spacing_X': 70,
                        'time_spacing_Xnew': 150
                    },
                    [([0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])],
                    ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
            )
    )
])
def mock_final_config(request):
    return request.param


def valid_config(input_config, output_config):
    if input_config[1]['scene']['benchmark'] != output_config['scene_params']['benchmark']:
        return False
    return True


@pytest.fixture
def mock_configs(mock_input_config, mock_final_config):
    if valid_config(mock_input_config, mock_final_config):
        return mock_input_config, mock_final_config
    else:
        pytest.skip("Skipping incompatible combination")


@pytest.fixture
def mock_parameter_loader():
    parameter_loader = ParameterLoader()
    return parameter_loader


@pytest.fixture
def mock_parameter_loader_with_yaml_config(test_params_path):
    parameter_loader = ParameterLoader()
    parameter_loader.initialize(file_path=test_params_path)
    return parameter_loader


@pytest.fixture
def mock_simulation_thread():
    simulation_thread = SimulationThread()

    # Return the simulation thread
    return simulation_thread


@pytest.fixture
def mock_simulation(mock_input_config):
    parameter_loader = initialized_parameter_loader(mock_input_config)
    simulation = initialized_simulation(parameter_loader)
    return parameter_loader, simulation


@pytest.fixture
def mock_robot(mock_simulation):
    parameter_loader, simulation = mock_simulation

    robot = Robot(parameter_loader, simulation)

    robot.initialise(default_robot_pos_and_orn=parameter_loader.scene_params['robot_pos_and_orn'],
                     joint_names=parameter_loader.robot_params['joint_names'],
                     default_pose=parameter_loader.robot_params['default_pose'],
                     benchmark=parameter_loader.scene_params['benchmark'])

    yield robot, mock_simulation


@pytest.fixture
def mock_sampler(mock_robot):
    robot, mock_simulation = mock_robot
    parameter_loader, simulation = mock_simulation

    sampler = Sampler(parameter_loader, robot)

    yield robot, sampler, mock_simulation


@pytest.fixture
def mock_simulation_manager_with_parameter_loader_and_simulation(mock_parameter_loader_with_yaml_config):
    simulation = initialized_simulation(mock_parameter_loader_with_yaml_config)
    simulation_manager = SimulationManager(parameter_loader=mock_parameter_loader_with_yaml_config,
                                           simulation=simulation)
    return simulation_manager


@pytest.fixture
def mock_simulation_manager_with_yaml_config(test_params_path):
    simulation_manager = SimulationManager(file_path=test_params_path)
    return simulation_manager


@pytest.fixture
def mock_bare_scene(mock_simulation_thread):
    mock_simulation_thread.initialize({'visuals': False})

    scene = Scene(client=mock_simulation_thread.client)
    return scene
