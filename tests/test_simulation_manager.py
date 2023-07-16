import numpy as np

from gpflow_vgpmp.utils.sdf_utils import SignedDistanceField
from gpflow_vgpmp.utils.simulation_manager import SimulationManager


def assert_everything_has_been_initialized_properly(simulation_manager, parameter_loader):
    assert simulation_manager._config.is_initialized
    assert simulation_manager.simulation.is_initialized
    assert simulation_manager.scene.is_initialized
    assert simulation_manager.robot.is_initialized

    if parameter_loader.scene_params['benchmark']:
        robot_position = [0.0, 0.0, 0.0]
        robot_orientation = [0.0, 0.0, -1.0, 0.0]
    else:
        robot_position = [0.0, 0.0, 0.0]
        robot_orientation = [0.0, 0.0, 0.0, 1.0]

    # robot_position = [0.0, 0.0, 0.0]
    # robot_orientation = [0.0, 0.0, 0.0, 1.0]
    scene_position = [-0.2, 0.0, 0.08]
    scene_orientation = [0.0, 0.0, 0.0, 1.0]

    assert simulation_manager.robot.position == robot_position
    assert simulation_manager.robot.orientation == robot_orientation
    assert simulation_manager.scene.position == scene_position
    assert simulation_manager.scene.orientation == scene_orientation

    sdf = SignedDistanceField.from_sdf(parameter_loader.params['scene_params']["sdf_path"])

    assert np.alltrue(simulation_manager.sdf.data == sdf.data)
    assert simulation_manager.scene.get_num_objects() == 2



# integration tests
def test_initialize_manager_yaml_config(mock_simulation_manager_with_yaml_config,
                                        mock_parameter_loader_with_yaml_config):
    assert_everything_has_been_initialized_properly(mock_simulation_manager_with_yaml_config, mock_parameter_loader_with_yaml_config)
    mock_simulation_manager_with_yaml_config.simulation.stop_simulation_thread()

def test_initialize_manager_parameter_loader_and_simulation(mock_sampler):
    robot, sampler, mock_simulation = mock_sampler
    parameter_loader, simulation = mock_simulation

    simulation_manager = SimulationManager(parameter_loader=parameter_loader,
                                           simulation=simulation)

    assert_everything_has_been_initialized_properly(simulation_manager,
                                                    parameter_loader)

    simulation_manager = SimulationManager(parameter_loader=parameter_loader,
                                             simulation=simulation,
                                                robot=robot)

    assert_everything_has_been_initialized_properly(simulation_manager,
                                                    parameter_loader)

    simulation_manager = SimulationManager(parameter_loader=parameter_loader,
                                                simulation=simulation,
                                                sampler=sampler)

    assert_everything_has_been_initialized_properly(simulation_manager,
                                                    parameter_loader)

    simulation_manager = SimulationManager(parameter_loader=parameter_loader,
                                                simulation=simulation,
                                                robot=robot,
                                                sampler=sampler)

    assert_everything_has_been_initialized_properly(simulation_manager,
                                                    parameter_loader)

    simulation_manager.simulation.stop_simulation_thread()
