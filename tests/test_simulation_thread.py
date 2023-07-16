import asyncio
import pytest
from unittest.mock import MagicMock
import pybullet as p

@pytest.mark.asyncio
async def test_run(mock_input_config, mock_simulation_thread):

    # Create a mock stop_event
    mock_simulation_thread.stop_event = MagicMock()
    mock_simulation_thread.initialize(mock_input_config[-1]['graphics'])

    # mock keys dictionary
    keys = {p.B3G_RETURN: p.KEY_WAS_TRIGGERED}

    # Mock the p.getKeyboardEvents() function
    p.getKeyboardEvents = MagicMock(return_value=keys)

    # Create a mock result_queue
    result_queue = MagicMock()
    mock_simulation_thread.result_queue = result_queue

    await mock_simulation_thread.run()
    mock_simulation_thread.stop()

    assert mock_simulation_thread.stop_event.set.call_count == 2  # once in run() and once in stop()

    # Ensure that the result_queue.put() method is called
    result_queue.put.assert_called_once_with(keys.keys())


@pytest.mark.asyncio
async def test_await_key_press(mock_simulation_thread, mock_input_config):
    mock_simulation_thread.initialize(mock_input_config[-1]['graphics'])

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

    mock_simulation_thread.stop()

    # Check the yielded key press events
    expected_key_presses = [
        (keys.keys(), False),
        (keys.keys(), True)
    ]

    for expected_press, actual_press in zip(expected_key_presses, key_presses):
        assert expected_press == actual_press


@pytest.mark.asyncio
async def test_wait_key_press(mock_simulation_thread, mock_input_config):
    # Create a mock stop_event
    mock_simulation_thread.stop_event = MagicMock()
    mock_simulation_thread.initialize(mock_input_config[-1]['graphics'])

    # Create a mock keys dictionary
    keys = {p.B3G_RETURN: p.KEY_WAS_TRIGGERED}

    # Mock the p.getKeyboardEvents() function
    p.getKeyboardEvents = MagicMock(return_value=keys)

    # Create a mock result_queue
    result_queue = MagicMock()

    # Assign the mock result_queue to the simulation_thread
    mock_simulation_thread.result_queue = result_queue

    # Use a context manager to handle exceptions and ensure the loop is properly closed
    # Call the wait_key_press method
    await mock_simulation_thread.wait_key_press()

    mock_simulation_thread.stop_event.set.assert_called_once()
    assert mock_simulation_thread.stop_event.is_set()

    # Ensure that p.getKeyboardEvents() is called
    p.getKeyboardEvents.assert_called_once()

    # Ensure that the result_queue.put is called with the correct key
    result_queue.put.assert_called_once_with(keys.keys())

    mock_simulation_thread.stop()


def test_initialize_thread(mock_simulation_thread, mock_input_config):
    mock_simulation_thread.thread_ready_event = MagicMock()
    # # Mock the pybullet methods and functions

    # Call the initialize() method
    mock_simulation_thread.initialize(mock_input_config[-1]['graphics'])
    # Assertions
    assert mock_simulation_thread.client == 0  # Ensure the client ID is set correctly
    mock_simulation_thread.thread_ready_event.set.assert_called_once()  # Ensure thread_ready_event.set() is called

    mock_simulation_thread.stop()


def test_stop_thread(mock_simulation_thread, mock_input_config):
    mock_simulation_thread.stop_event = MagicMock()
    mock_simulation_thread.initialize(mock_input_config[-1]['graphics'])
    mock_simulation_thread.thread_ready_event.wait()

    # Call the stop_thread() method
    mock_simulation_thread.stop()

    assert mock_simulation_thread.client is None  # Ensure the client ID is set to None
    assert mock_simulation_thread.stop_event.is_set()  # Ensure stop_event.is_set() is called


@pytest.mark.asyncio
async def test_check_connection(mock_simulation_thread, mock_input_config):
    mock_simulation_thread.stop_event = MagicMock()
    mock_simulation_thread.initialize(mock_input_config[-1]['graphics'])

    p = MagicMock()
    p.getConnectionInfo = MagicMock(return_value={'isConnected': 0})

    assert mock_simulation_thread.thread_ready_event.is_set()

    # Call the check_connection method
    await mock_simulation_thread.check_connection()

    assert mock_simulation_thread.stop_event.is_set()  # Ensure stop_event.is_set() is called

    mock_simulation_thread.stop()
