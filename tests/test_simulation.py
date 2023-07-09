import asyncio
from pathlib import Path
import pybullet as p
import pytest
from unittest.mock import MagicMock
from gpflow_vgpmp.utils.miscellaneous import get_root_package_path


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


def test_parameter_loader_load_parameter_file(mock_parameter_loader_with_paths, mock_input_config, mock_final_config):

    # Test loading a parameter file
    mock_parameter_loader_with_paths.initialize(params=mock_input_config)

    sensitive_keys = ['object_path', 'urdf_path', 'sdf_path', 'environment_path']
    # Ensure the initial params are set correctly

    is_equal = compare_dicts(mock_parameter_loader_with_paths.params['robot_params'], mock_final_config['robot_params'],
                             sensitive_keys)
    assert is_equal
    is_equal = compare_dicts(mock_parameter_loader_with_paths.params['scene_params'], mock_final_config['scene_params'],
                             sensitive_keys)
    assert is_equal
    is_equal = compare_dicts(mock_parameter_loader_with_paths.params['trainable_params'], mock_final_config['trainable_params'],
                             [])
    assert is_equal
    is_equal = compare_dicts(mock_parameter_loader_with_paths.params['graphics_params'], mock_final_config['graphics_params'], [])
    assert is_equal
    is_equal = compare_dicts(mock_parameter_loader_with_paths.params['planner_params'], mock_final_config['planner_params'], [])
    assert is_equal


@pytest.mark.asyncio
async def test_initialize_thread(mock_simulation_thread):
    mock_simulation_thread.thread_ready_event = MagicMock()
    # # Mock the pybullet methods and functions
    p = MagicMock()
    p.connect.return_value = 0  # Return a mock client ID

    # Call the initialize() method
    mock_simulation_thread.initialize()

    # Assertions
    assert mock_simulation_thread.client == 0  # Ensure the client ID is set correctly
    mock_simulation_thread.thread_ready_event.set.assert_called_once()  # Ensure thread_ready_event.set() is called


@pytest.mark.asyncio
async def test_stop_thread(mock_simulation_thread):
    mock_simulation_thread.stop_event = MagicMock()
    mock_simulation_thread.initialize()
    mock_simulation_thread.thread_ready_event.wait()

    # Call the stop_thread() method
    mock_simulation_thread.stop()

    assert mock_simulation_thread.client is None  # Ensure the client ID is set to None
    assert mock_simulation_thread.stop_event.is_set()  # Ensure stop_event.is_set() is called


@pytest.mark.asyncio
async def test_check_connection(mock_simulation_thread):
    mock_simulation_thread.stop_event = MagicMock()
    mock_simulation_thread.initialize()

    assert mock_simulation_thread.thread_ready_event.is_set()

    p.disconnect(mock_simulation_thread.client)  # Disconnect the client

    # Call the check_connection method
    await mock_simulation_thread.check_connection()

    assert mock_simulation_thread.stop_event.is_set()  # Ensure stop_event.is_set() is called


@pytest.mark.asyncio
async def test_await_key_press(mock_simulation_thread):
    mock_simulation_thread.initialize()

    # Create a mock stop_event
    mock_simulation_thread.stop_event = MagicMock()
    mock_simulation_thread.stop_event.is_set.return_value = False

    # Mock the p.getKeyboardEvents() function
    keys = {
        27: p.KEY_WAS_TRIGGERED,
        32: p.KEY_WAS_RELEASED,
        p.B3G_RETURN: p.KEY_WAS_TRIGGERED
    }
    p.getKeyboardEvents = MagicMock(return_value=keys)

    # Call the await_key_press method
    key_presses = []
    async for key_press in mock_simulation_thread.await_key_press():
        key_presses.append(key_press)
        if len(key_presses) == 2:
            break

    # Check the yielded key press events
    expected_key_presses = [
        (keys.keys(), False),
        (keys.keys(), True)
    ]

    for expected_press, actual_press in zip(expected_key_presses, key_presses):
        assert expected_press == actual_press


@pytest.mark.asyncio
async def test_wait_key_press(mock_simulation_thread):
    # Create a mock stop_event
    mock_simulation_thread.stop_event = MagicMock()
    mock_simulation_thread.initialize()

    # Create a mock keys dictionary
    keys = {p.B3G_RETURN: p.KEY_WAS_TRIGGERED}

    # Mock the p.getKeyboardEvents() function
    p.getKeyboardEvents = MagicMock(return_value=keys)

    # Create a mock result_queue
    result_queue = MagicMock()

    # Assign the mock result_queue to the simulation_thread
    mock_simulation_thread.result_queue = result_queue

    # Use a context manager to handle exceptions and ensure the loop is properly closed
    try:
        # Call the wait_key_press method
        await mock_simulation_thread.wait_key_press()

        mock_simulation_thread.stop_event.set.assert_called_once()
        assert mock_simulation_thread.stop_event.is_set()

        # Ensure that p.getKeyboardEvents() is called
        p.getKeyboardEvents.assert_called_once()

        # Ensure that the result_queue.put is called with the correct key
        result_queue.put.assert_called_once_with(keys.keys())
    finally:
        # Clean up any resources and cancel the pending tasks
        asyncio.get_running_loop().stop()


@pytest.mark.asyncio
async def test_run(mock_input_config, mock_simulation_thread):
    # Create a mock stop_event
    mock_simulation_thread.stop_event = MagicMock()
    mock_simulation_thread.initialize()

    # Create a mock keys dictionary
    keys = {p.B3G_RETURN: p.KEY_WAS_TRIGGERED}

    # Mock the p.getKeyboardEvents() function
    p.getKeyboardEvents = MagicMock(return_value=keys)

    # Create a mock result_queue
    result_queue = MagicMock()

    # Assign the mock result_queue to the simulation_thread
    mock_simulation_thread.result_queue = result_queue

    # Use a context manager to handle exceptions and ensure the loop is properly closed
    try:
        await mock_simulation_thread.run()

        # Ensure that stop_event.is_set() is called
        mock_simulation_thread.stop_event.set.assert_called_once()
    finally:
        # Clean up any resources and cancel the pending tasks
        asyncio.get_running_loop().stop()


@pytest.mark.asyncio
async def test_initialize_simulation(mock_simulation):
    mock_simulation.initialize()

    assert mock_simulation.simulation_thread.client == 0
    assert mock_simulation.simulation_thread.thread_ready_event.is_set()


@pytest.mark.asyncio
async def test_stop_simulation(mock_simulation):
    mock_simulation.initialize()
    mock_simulation.simulation_thread.thread_ready_event.wait()

    mock_simulation.stop_simulation_thread()

    assert mock_simulation.simulation_thread.client is None
    assert mock_simulation.simulation_thread.stop_event.is_set()
