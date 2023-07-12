
# integration tests
def test_initialize_manager(mock_simulation_manager, mock_final_config):

    if mock_final_config['scene_params']['benchmark']:
        position = [0.0, 0.0, 0.0]
        orientation = [0.0, 0.0, -1.0, 0.0]
    else:
        position = [0.0, 0.0, 0.0]
        orientation = [0.0, 0.0, 0.0, 1.0]
    print(mock_final_config['scene_params']['robot_pos_and_orn'])
    assert mock_simulation_manager.robot.position == position
    assert mock_simulation_manager.robot.orientation == orientation



