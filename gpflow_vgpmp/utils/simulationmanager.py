from pathlib import Path
import tensorflow as tf
from gpflow_vgpmp.utils.miscellaneous import get_root_package_path
from gpflow_vgpmp.utils.robot import Robot
from gpflow_vgpmp.utils.sdf_utils import SignedDensityField
from gpflow_vgpmp.utils.simulation import Simulation, Scene
from gpflow_vgpmp.utils.parameter_loader import ParameterLoader

# ---------------Exports
__all__ = 'simulator'


class SimulationManager:
    def __init__(self):
        # Lazy initialization
        self.robot = None
        self.sdf = None
        self.scene = None
        self.simulation = None
        self._config = None
        self.client = None

    @property
    def config(self) -> dict:
        assert self._config is not None, "Parameter Loader must be initialized before it can be accessed"
        return self._config.params

    def initialize(self, parameter_file_path=None):
        self._config = ParameterLoader(parameter_file_path)
        self.initialize_parameter_loader()
        self.simulation = Simulation(self.config['graphics_params'])
        self.initialize_simulation()
        self.scene = Scene(self.config['scene_params'])
        self.initialize_scene()
        self.sdf = SignedDensityField.from_sdf(self.config['scene_params']["sdf_path"])
        self.robot = Robot(self.config['robot_params'], self.simulation.simulation_thread.client)
        self.initialize_robot()

    def initialize_parameter_loader(self):
        self._config.initialize()

    def initialize_simulation(self):
        self.simulation.initialize()

    def initialize_scene(self):
        self.assert_backend_connection_is_alive()
        self.scene.initialize(self.simulation.simulation_thread.client)

    def initialize_robot(self):
        self.assert_backend_connection_is_alive()
        robot_pos_and_orn = self.config['scene_params']['robot_pos_and_orn']
        joint_names = self.config['robot_params']['joint_names']
        default_pose = self.config['robot_params']['default_pose']
        benchmark = self.config['scene_params']['benchmark']

        self.robot.initialise(default_robot_pos_and_orn=robot_pos_and_orn,
                              joint_names=joint_names,
                              default_pose=default_pose,
                              benchmark=benchmark)

    def loop(self, planner=None):
        exit = False
        while not exit:
            action = input("Enter action: ")
            if action == "q":
                exit = True
            elif action == 'c':
                print(f"Current config is :{self.robot.get_current_joint_config()}")
            elif action == 'sdf':
                if planner is not None:
                    self.get_rt_sdf_grad(planner)
                else:
                    print("There was no planner given")
            elif action == 'fk':
                if planner is not None:
                    joints = self.robot.get_current_joint_config()
                    tf.print(planner.debug_likelihood(tf.reshape(joints, (1, 1, 7))))
                else:
                    print("There was no planner given")
            else:
                print(f"Current config is :{self.robot.get_current_joint_config(int(action))}")

    def get_rt_sdf_grad(self, planner):
        """
        Get the signed distance gradient of the current robot configuration and print it
        """
        joints = self.robot.get_current_joint_config().reshape(7, 1)
        position = planner.likelihood.sampler._fk_cost(joints)
        print(planner.likelihood._signed_distance_grad(position))

    def assert_backend_connection_is_alive(self):
        assert self.simulation.simulation_thread.client is not None, "Client must be initialized before scene"
        assert self.simulation.simulation_thread.thread_ready_event.is_set(), "Client must be initialized before scene"

if __name__ == "__main__":
    parameter_file_path = Path(get_root_package_path()) / "parameters.yaml"
    env = SimulationManager()
    env.initialize(parameter_file_path)
