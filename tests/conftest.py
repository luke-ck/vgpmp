from unittest.mock import MagicMock
import pytest
import numpy as np
from gpflow_vgpmp.utils.simulation import Simulation, SimulationThread
from gpflow_vgpmp.utils.parameter_loader import ParameterLoader
from gpflow_vgpmp.utils.simulationmanager import SimulationManager


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
                "environment_file_name": "industrial",
                "sdf_file_name": "industrial_vgpmp",
                "objects": [],
                "objects_position": [],
                "objects_orientation": [],
                "problemset": "industrial",
                "sdf_name": "industrial_vgpmp",
                "benchmark": False,
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
                            "lengthscales": [5.0, 5.0, 5.0, 2.0, 5.0, 5.0, 5.0],
                            "variance": 0.25,
                            "alpha": 100,
                            "num_samples": 7,
                            "num_inducing": 24,
                            "learning_rate": 0.09,
                            "num_steps": 130,
                            "time_spacing_X": 70,
                            "time_spacing_Xnew": 150
                        }
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


@pytest.fixture
def mock_final_config():
    # Create a mock configuration dictionary
    return {
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
            'no_frames_for_spheres': 5,
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
            'benchmark': False,
            'object_path': [],
            'queries': [([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])],
            'robot_pos_and_orn': ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
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
            'visuals': True,
            'quality': 'high',
            'GUI': True,
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
    parameter_loader = ParameterLoader('tests/test_params.yaml')
    robot_params, scene_params, trainable_params, graphic_params = mock_input_config
    parameter_loader.scene_params = scene_params["scene"]
    parameter_loader.robot_params = robot_params["robot"]
    parameter_loader.trainable_params = trainable_params["trainable_params"]
    parameter_loader.graphics_params = graphic_params["graphics"]

    parameter_loader.set_params = MagicMock(return_value=mock_final_config)
    return parameter_loader


@pytest.fixture
def mock_simulation_thread(mock_input_config):
    graphic_params = mock_input_config[-1]['graphics']

    # Create a mock simulation thread
    simulation_thread = SimulationThread(graphic_params)
    yield simulation_thread


@pytest.fixture
def mock_simulation(mock_input_config, mock_simulation_thread):
    graphic_params = mock_input_config[-1]['graphics']
    # Create a mock simulation with the mock simulation thread
    simulation = Simulation(graphic_params)
    simulation.simulation_thread = mock_simulation_thread

    yield simulation


@pytest.fixture
def mock_simulation_manager(mock_input_config, mock_parameter_loader):
    robot_params, scene_params, trainable_params, graphic_params = mock_input_config
    # Create a mock simulator with the mock simulation
    simulation_manager = SimulationManager('tests/test_params.yaml')
    simulation_manager._config = mock_parameter_loader

    yield simulation_manager
