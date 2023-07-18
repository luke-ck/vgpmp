import asyncio
import threading
from pathlib import Path
import pybullet as p
import pytest
from unittest.mock import MagicMock
from gpflow_vgpmp.utils.miscellaneous import get_root_package_path


#
def compare_dicts(dict1, dict2, ignore_keys):
    # Create copies of the dictionaries to avoid modifying the original ones
    dict1_copy = dict1.copy()
    dict2_copy = dict2.copy()

    # Remove the keys containing sensitive information from the copies
    for key in ignore_keys:
        dict1_copy.pop(key, None)
        dict2_copy.pop(key, None)

    # Perform the equality check
    return dict1_copy == dict2_copy


def test_parameter_loader_load_parameter_file(mock_parameter_loader, mock_configs):
    mock_input_config, mock_output_config = mock_configs

    # Test loading a parameter file
    mock_parameter_loader.initialize(params=mock_input_config)

    sensitive_keys = ['objects_path', 'urdf_path', 'sdf_path', 'environment_path']
    # Ensure the initial params are set correctly

    is_equal = compare_dicts(mock_parameter_loader.params['robot_params'],
                             mock_output_config['robot_params'],
                             sensitive_keys)
    assert is_equal
    is_equal = compare_dicts(mock_parameter_loader.params['scene_params'],
                             mock_output_config['scene_params'],
                             sensitive_keys)
    assert is_equal
    is_equal = compare_dicts(mock_parameter_loader.params['trainable_params'],
                             mock_output_config['trainable_params'],
                             [])
    assert is_equal
    is_equal = compare_dicts(mock_parameter_loader.params['graphics_params'],
                             mock_output_config['graphics_params'], [])
    assert is_equal
    is_equal = compare_dicts(mock_parameter_loader.params['planner_params'],
                             mock_output_config['planner_params'], [])
    assert is_equal


def test_initialize_simulation(mock_simulation):
    # shared_mock_simulation.initialize()
    _, simulation = mock_simulation
    simulation.simulation_thread.thread_ready_event.wait()

    assert simulation.simulation_thread.client == 0
    assert simulation.simulation_thread.thread_ready_event.is_set()

    simulation.stop_simulation_thread()


def test_stop_simulation(mock_simulation):
    # mock_simulation.initialize()
    _, simulation = mock_simulation
    simulation.simulation_thread.thread_ready_event.wait()

    simulation.stop_simulation_thread()

    assert simulation.simulation_thread.client is None
    assert simulation.simulation_thread.stop_event.is_set()
