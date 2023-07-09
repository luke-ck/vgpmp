import pybullet as p
import pytest
from unittest.mock import MagicMock


# integration tests
@pytest.mark.asyncio
async def test_initialize_robot(mock_simulation_manager, mock_final_config):
    pass
    # assert mock_simulation_manager.robot.robot_model is 0
    #
    # position = [0, 0, 0]
    # orientation = [0, 0, 0, 1]
    #
    # assert mock_simulation_manager.robot.position == position
    # assert mock_simulation_manager.robot.orientation == orientation



