import warnings
from pathlib import Path
import tensorflow as tf

from gpflow_vgpmp.models.vgpmp import VGPMP
from gpflow_vgpmp.utils.miscellaneous import get_root_package_path
from gpflow_vgpmp.utils.robot import Robot
from gpflow_vgpmp.utils.sampler import Sampler
from gpflow_vgpmp.utils.sdf_utils import SignedDistanceField
from gpflow_vgpmp.utils.simulation import Simulation, Scene
from gpflow_vgpmp.utils.parameter_loader import ParameterLoader

# ---------------Exports
__all__ = 'simulator'


def assert_component_is_initialized(component):
    assert component.is_initialized, f"{component.__class__.__name__} must be initialized before it can be used"


class SimulationManager:
    """
    This class is responsible for managing the interaction between the different components of the simulator.
    It is responsible for initializing the simulation, the scene, the robot, the sdf and the sampler.
    One can either initialize the components manually (preferable for testing) or by loading a parameter file during
    the initialization of the simulation manager (preferable for running the simulator).
    """

    def __init__(self,
                 parameter_loader: ParameterLoader = None,
                 simulation: Simulation = None,
                 scene: Scene = None,
                 robot: Robot = None,
                 sdf: SignedDistanceField = None,
                 sampler: Sampler = None,
                 parameter_file_path=None):
        """
        Either initialize the components manually or by loading a parameter file during the initialization of the
        simulation manager. At the very least, the parameter loader and the simulation must be initialized.
        """
        if parameter_file_path is not None:
            self._config = ParameterLoader()
            self.initialize_parameter_loader(parameter_file_path)
            self.simulation = Simulation(self._config)
            self.initialize_simulation()
            assert_component_is_initialized(self.simulation)
            self.scene = Scene(self._config)
            self.initialize_scene()
            self.robot = Robot(self._config, self.simulation)
            self.initialize_robot()
            assert_component_is_initialized(self.robot)
            self.sampler = Sampler(self._config, self.robot)
            self.sdf = SignedDistanceField.from_sdf(self.config['scene_params']["sdf_path"])
        else:
            assert parameter_loader is not None and parameter_loader.is_initialized, "Parameter Loader must be " \
                                                                                     "initialized if no parameter " \
                                                                                     "file is passed"
            assert simulation is not None and simulation.is_initialized, "Simulation must have been initialized if no " \
                                                                         "parameter file is passed "
            self._config = parameter_loader
            self.simulation = simulation
            self.assert_backend_connection_is_alive()
            self.client = self.simulation.simulation_thread.client

            if scene is not None:
                if scene.is_initialized:
                    self.scene = scene
                else:
                    warnings.warn("You passed a scene that is not initialized. Initializing scene with default "
                                  "parameters")
            else:
                self.scene = Scene(self._config)
                self.initialize_scene()

            if robot is not None:
                if robot.is_initialized:
                    self.robot = robot
                else:
                    warnings.warn("You passed a robot that is not initialized. Initializing robot with default "
                                  "parameters")
            else:
                self.robot = Robot(self._config, self.simulation)
                self.initialize_robot()

            if sdf is not None:
                self.sdf = sdf
            else:
                self.sdf = SignedDistanceField.from_sdf(self.config['scene_params']["sdf_path"])
            if sampler is not None:
                assert self.robot is not None and self.robot.is_initialized, "Robot must be initialized before a " \
                                                                             "sampler can be added"
                self.sampler = sampler
            else:
                self.sampler = Sampler(self._config, self.robot)

    @property
    def config(self) -> dict:
        assert self._config is not None, "Parameter Loader must be initialized before it can be accessed"
        return self._config.params

    # def initialize(self):
    #     """
    #     Initializes the simulation manager by initializing the parameter loader, the simulation, the scene, the robot,
    #     the sdf and the sampler. If a parameter file is passed, the parameter loader is initialized with the parameter
    #     file. Otherwise, the parameter loader is initialized with the default parameter file.
    #     """
    #     if self._config is None:
    #         self._config = ParameterLoader()
    #         self.initialize_parameter_loader(parameter_file_path)
    #
    #         self.simulation = Simulation(self.config['graphics_params'])
    #         self.initialize_simulation()
    #         self.assert_component_is_initialized(self.simulation)
    #
    #         self.scene = Scene(self.config['scene_params'])
    #         self.initialize_scene()
    #         self.assert_component_is_initialized(self.scene)
    #
    #         self.sdf = SignedDistanceField.from_sdf(self.config['scene_params']["sdf_path"])
    #
    #         self.robot = Robot(self.config['robot_params'], self.simulation.simulation_thread.client)
    #         self.initialize_robot()
    #         self.assert_component_is_initialized(self.robot)
    #         self.sampler = Sampler(self.config['robot_params'], self.robot.base_pose, self.robot.sphere_offsets)
    #     else:
    #         self.assert_component_is_initialized(self._config)

    def initialize_parameter_loader(self, file_path: Path = None, params: dict = None):
        self._config.initialize(file_path=file_path, params=params)

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
        position = planner.likelihood.sampler.forward_kinematics_cost(joints)
        print(planner.likelihood._signed_distance_grad(position))

    def assert_backend_connection_is_alive(self):
        assert self.simulation.simulation_thread.client is not None, "Connection to backend not established"
        assert self.simulation.simulation_thread.thread_ready_event.is_set(), "Simulation thread not ready"
        assert self.simulation.is_initialized, "Simulation must be initialized"


if __name__ == "__main__":
    parameter_file_path = Path(get_root_package_path()) / "parameters.yaml"
    env = SimulationManager(parameter_file_path=parameter_file_path)
    # env.initialize()

    y = tf.constant([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=tf.float64)
    planner = VGPMP.initialize(sdf=env.sdf,
                               robot=env.robot,
                               sampler=env.sampler,
                               query_states=y,
                               scene_offset=env.scene.position,
                               **env.config['planner_params'])
