import asyncio
import pybullet as p
import pytest
from unittest.mock import MagicMock
from gpflow_vgpmp.utils.simulation import SimulationThread


@pytest.mark.asyncio
async def test_initialize(mock_input_config):
    # Create a mock graphic_params dictionary
    graphic_params = mock_input_config[-1]['graphics']

    # Create an instance of the SimulationThread class
    simulation_thread = SimulationThread(graphic_params, thread_ready_event=MagicMock(), queue=MagicMock())

    # # Mock the pybullet methods and functions
    p = MagicMock()
    p.connect.return_value = 0  # Return a mock client ID

    # Call the initialize() method
    simulation_thread.initialize()

    # Assertions
    assert simulation_thread.client == 0  # Ensure the client ID is set correctly
    simulation_thread.thread_ready_event.set.assert_called_once()  # Ensure thread_ready_event.set() is called


@pytest.mark.asyncio
async def test_check_connection(mock_input_config):
    # Create a mock graphic_params dictionary
    graphic_params = mock_input_config[-1]['graphics']

    # Create an instance of the SimulationThread class
    simulation_thread = SimulationThread(graphic_params, thread_ready_event=MagicMock(), queue=MagicMock())
    simulation_thread.stop_event = MagicMock()
    simulation_thread.initialize()

    simulation_thread.thread_ready_event.is_set()

    p.disconnect(simulation_thread.client)  # Disconnect the client

    # Call the check_connection method
    await simulation_thread.check_connection()

    simulation_thread.stop_event.is_set()  # Ensure stop_event.is_set() is called


@pytest.mark.asyncio
async def test_await_key_press(mock_input_config):
    graphic_params = mock_input_config[-1]['graphics']

    # Create an instance of SimulationThread
    simulation_thread = SimulationThread(graphic_params, thread_ready_event=MagicMock(), queue=MagicMock())
    simulation_thread.initialize()

    # Create a mock stop_event
    simulation_thread.stop_event = MagicMock()
    simulation_thread.stop_event.is_set.return_value = False

    # Mock the p.getKeyboardEvents() function
    keys = {
            27: p.KEY_WAS_TRIGGERED,
            32: p.KEY_WAS_RELEASED,
            p.B3G_RETURN: p.KEY_WAS_TRIGGERED
            }
    p.getKeyboardEvents = MagicMock(return_value=keys)

    # Call the await_key_press method
    key_presses = []
    async for key_press in simulation_thread.await_key_press():
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
async def test_wait_key_press(mock_input_config):
    graphic_params = mock_input_config[-1]['graphics']

    # Create an instance of SimulationThread
    simulation_thread = SimulationThread(graphic_params, thread_ready_event=MagicMock(), queue=MagicMock())

    # Create a mock stop_event
    simulation_thread.stop_event = MagicMock()
    simulation_thread.stop_event.is_set.side_effect = [False, True]  # Simulate stop event being set
    simulation_thread.initialize()

    # Create a mock keys dictionary
    keys = {p.B3G_RETURN: p.KEY_WAS_TRIGGERED}

    # Mock the p.getKeyboardEvents() function
    p.getKeyboardEvents = MagicMock(return_value=keys)

    # Create a mock result_queue
    result_queue = MagicMock()

    # Assign the mock result_queue to the simulation_thread
    simulation_thread.result_queue = result_queue

    # Use a context manager to handle exceptions and ensure the loop is properly closed
    try:
        # Call the wait_key_press method
        await simulation_thread.wait_key_press()
        # Ensure that stop_event.is_set() is called
        simulation_thread.stop_event.set.assert_called_once()

        # Ensure that p.getKeyboardEvents() is called
        p.getKeyboardEvents.assert_called_once()

        # Ensure that the result_queue.put is called with the correct key
        result_queue.put.assert_called_once_with(keys.keys())
    finally:
        # Clean up any resources and cancel the pending tasks
        asyncio.get_running_loop().stop()


@pytest.mark.asyncio
async def test_run(mock_input_config):
    graphic_params = mock_input_config[-1]['graphics']

    # Create an instance of the SimulationThread class
    simulation_thread = SimulationThread(graphic_params, thread_ready_event=MagicMock(), queue=MagicMock())

    # Create a mock stop_event
    simulation_thread.stop_event = MagicMock()
    simulation_thread.initialize()

    # Create a mock keys dictionary
    keys = {p.B3G_RETURN: p.KEY_WAS_TRIGGERED}

    # Mock the p.getKeyboardEvents() function
    p.getKeyboardEvents = MagicMock(return_value=keys)

    # Create a mock result_queue
    result_queue = MagicMock()

    # Assign the mock result_queue to the simulation_thread
    simulation_thread.result_queue = result_queue

    # Use a context manager to handle exceptions and ensure the loop is properly closed
    try:
        await simulation_thread.run()

        # Ensure that stop_event.is_set() is called
        simulation_thread.stop_event.set.assert_called_once()
    finally:
        # Clean up any resources and cancel the pending tasks
        asyncio.get_running_loop().stop()


def test_parameter_loader_load_parameter_file(mock_parameter_loader, mock_input_config, mock_final_config):
    # Test loading a parameter file
    parameter_file_path = "./test_params.yaml"

    # Ensure the initial params are set correctly
    assert mock_parameter_loader.params == mock_input_config

    # Call the load_parameter_file method
    mock_parameter_loader.load_parameter_file(parameter_file_path)

    # Verify that the set_params method is called with mock_input_config
    mock_parameter_loader.set_params.assert_called_with(mock_input_config)

    # Ensure that the params attribute is updated with mock_final_config
    assert mock_parameter_loader.params == mock_final_config

