import asyncio
import queue
import sys
import threading
from pathlib import Path
from typing import List, Optional, Tuple
import pkg_resources
import pybullet as p
import yaml
from pkg_resources import DistributionNotFound
from .bullet_object import BaseObject
from .miscellaneous import get_root_package_path

__all__ = 'simulation'


def get_bullet_key_from_value():
    """ Get the key from the value in the bullet dictionary """
    return {value: key for key, value in p.__dict__.items() if key.startswith("B3G")}


def load_yaml_config(scene_config):
    with open(scene_config, 'r') as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.constructor.ConstructorError as e:
            print(e)
    return config_dict


class ParameterLoader:
    """
    Load and extract data from a parameter files, which are used to configure the simulator. On start up, the loader
    configures the following parameters:
    - self.scene_params = parameters dealing with the scene (managing objects, sdf, the environment)
    - self.robot_params = parameters necessary to set up the robot (using robot specific config
    parameter file + general robot parameters found in the central config file)
    - self.graphic_params = visual parameters passed to SimulationThread to establish the pybullet connection
    - self.trainable_params = parameters that are used to configure the GP model (whether to train q_mu, q_sqrt, etc.)
    """

    def __init__(self, parameter_file_path=None):
        self._params = None
        self.trainable_params = None
        self.graphics_params = None
        self.robot_params = None
        self.scene_params = None
        root_package_path = Path(get_root_package_path())
        self.data_dir_path = root_package_path / 'data'
        if parameter_file_path is not None:
            parameter_file_path = root_package_path / parameter_file_path
        try:
            with open(root_package_path / 'requirements.txt', 'r') as req:
                pkg_resources.require(req)
        except DistributionNotFound:
            print('[Warning]: Missing packages ')
            sys.exit(
                '[EXIT]: System will exit, please install all requirements and run the simulator again')

        self.load_parameter_file(parameter_file_path)

    @property
    def params(self):
        return self._params

    def set_params(self, params):
        """ Load and extract data from parameter file """
        robot_params, scene_params, trainable_params, graphic_params = params
        self.scene_params = scene_params["scene"]
        self.robot_params = robot_params["robot"]
        self.trainable_params = trainable_params["trainable_params"]
        self.graphics_params = graphic_params["graphics"]

        self.get_robot_config(self.robot_params)
        self.get_scene_config(self.scene_params)

        self._params = {'robot_params': self.robot_params,
                        'scene_params': self.scene_params,
                        'trainable_params': self.trainable_params,
                        'graphics_params': self.graphics_params}

    def get_robot_config(self, robot_params):
        """ Load robot configuration from parameter file which is found together with the rest of the robot files
            The robot configuration is saved in the robot_params dict.
            The general idea is this: general robot parameters are saved in the main parameter file,
            and robot specific parameters (FK related, joint limits, spheres and so on)
            are saved in the robot specific parameter file.
         """
        robot_name = robot_params["robot_name"]  # specify which robot to load
        robots_data_dir_path = self.data_dir_path / "robots"
        robot_path = robots_data_dir_path / robot_name
        robot_config = robot_path / "config.yaml"
        config_dict = load_yaml_config(robot_config)
        config_dict["urdf_path"] = robot_path / config_dict["path"]

        # concatenate the robot parameters with the robot config
        # if there are any duplicates, robot parameters have precedence
        self.robot_params = {**config_dict, **robot_params}

    def get_scene_config(self, scene_params):
        """Load scene paths require parameter_file_path for SDF and URDF files """

        problemset = scene_params["problemset"]
        environment_name = scene_params["environment_name"]
        sdf_name = scene_params["sdf_name"]

        scene_params["object_path"] = []
        if scene_params["objects"] is not None and scene_params["objects"] is not []:
            objects_data_dir_path = self.data_dir_path / "objects"
            for obj in scene_params["objects"]:
                object_path = objects_data_dir_path / obj
                scene_params["object_path"].append(object_path)

        environment_path, sdf_path = self.get_assets_path(problemset, environment_name, sdf_name)

        scene_params["sdf_path"] = sdf_path
        scene_params["environment_path"] = environment_path

        self.scene_params = scene_params

    def get_assets_path(self, problemset: str, environment_name: str, sdf_name: str) -> Tuple[Path, Path]:
        scenes_data_dir_path = Path(self.data_dir_path) / "scenes"
        scene_path = scenes_data_dir_path / problemset / (environment_name + ".urdf")
        sdf_path = scenes_data_dir_path / problemset / (sdf_name + ".sdf")
        assert scene_path.exists(), f"Scene file {scene_path} does not exist"
        assert sdf_path.exists(), f"SDF file {sdf_path} does not exist"
        return scene_path, sdf_path

    def load_parameter_file(self, parameter_file_path: Path):
        """ Load parameter file """
        try:
            with open(parameter_file_path, 'r') as stream:
                params = yaml.safe_load(stream)
        except FileNotFoundError:
            print("[Error]: Parameters file could not be found")
            sys.exit('[EXIT]: System will exit, please provide a parameter file and try again')
        except yaml.constructor.ConstructorError as e:
            print(e)
        else:
            self.set_params(params)


class SimulationThread(threading.Thread):
    def __init__(self, graphic_params: dict):
        super().__init__()
        self.graphic_params = graphic_params
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.thread_ready_event = threading.Event()
        self.client = None

    async def check_connection(self):
        assert self.client is not None, "Client is not initialized"
        while not self.stop_event.is_set():
            if p.getConnectionInfo(self.client)['isConnected'] == 0:
                self.stop_event.set()
                return
            if self.stop_event.is_set():
                return
            await asyncio.sleep(0.01)
        self.client = None

    async def wait_key_press(self):
        async for key, is_return_pressed in self.await_key_press():
            self.result_queue.put(key)
            if is_return_pressed:
                self.stop_event.set()
                return

    async def await_key_press(self):
        while True:
            keys = p.getKeyboardEvents()
            if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
                yield keys.keys(), False
                return

            if p.B3G_RETURN in keys and keys[p.B3G_RETURN] & p.KEY_WAS_TRIGGERED:
                yield keys.keys(), True
                return

            if self.stop_event.is_set():
                yield "STOP", True
                return

            await asyncio.sleep(0.01)

    async def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        self.initialize()

        tasks = [self.check_connection(), self.wait_key_press()]
        try:
            await asyncio.gather(*[asyncio.create_task(task) for task in tasks])
        except asyncio.CancelledError:
            pass
        finally:
            loop.close()

    def initialize(self):

        gravity_constant = -9.81

        if self.graphic_params["visuals"]:
            self.client = p.connect(p.GUI, options="--width=960 --height=540")
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.setRealTimeSimulation(enableRealTimeSimulation=1)
            p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)
        else:
            self.client = p.connect(p.DIRECT)
        p.resetSimulation()
        p.setGravity(0, 0, gravity_constant)

        self.thread_ready_event.set()  # Notify the main thread that the simulation thread is ready


class Simulation:
    def __init__(self, graphic_params):
        self.result_queue = None
        self.simulation_thread = None
        self.scene = None
        self.result_queue = queue.Queue()
        self.graphic_params = graphic_params
        self.thread_ready_event = threading.Event()  # Create a new event object
        # self.start_engine()
        self.start_simulation_thread()
        # Wait for the SimulationThread instance to finish initializing
        self.thread_ready_event.wait()

    def start_simulation_thread(self):
        self.simulation_thread = SimulationThread(self.graphic_params)
        self.simulation_thread.start()
        print("Simulation thread started")

    def stop_simulation_thread(self):
        self.simulation_thread.stop()
        self.simulation_thread.join()
        print("Simulation thread stopped")

    def check_events(self):
        """ TODO: do stuff with the simulation thread"""
        # queue_result = self.result_queue.get().result()
        # queue_size = self.result_queue.qsize()
        # print("Queue size: ", queue_size)
        # print("Queue result: ", queue_result)
        # for key in queue_result:
        #     if key == p.B3G_RETURN:
        #         self.simulation_thread.stop()
        #         self.simulation_thread.join()
        #         print("Simulation thread stopped")
        #     else:
        #         print("Something went wrong with the simulation thread")
        pass

    def check_simulation_thread_health(self):
        """ check if the simulation thread is still alive """
        return self.simulation_thread.is_alive()


class Scene:
    """
    Class for managing a scene of BaseObjects.
    """

    def __init__(self, config: dict):
        """
        Create a new Scene instance.
        """
        self.config = config
        self.objects = []
        self.initialize()

    def add_object(self, name: str, path: Optional[str], position: Optional[List], orientation: Optional[List]):
        """
        Add a new BaseObject instance to the scene.
        """
        path = Path(path) if type(path) == str else path  # Convert path to Path object if it is a string
        obj = BaseObject(name=name, path=path, position=position, orientation=orientation)
        self.objects.append(obj)

    def remove_object(self, obj: BaseObject):
        """
        Remove a BaseObject instance from the scene.
        """
        obj.remove()
        self.objects.remove(obj)

    def get_object_by_name(self, name: str) -> Optional[BaseObject]:
        """
        Get a BaseObject instance from the scene by name.

        Args:
            name(str): The name of the BaseObject instance to retrieve.

        Returns:
            The BaseObject instance with the specified name, or None if not found.
        """
        for obj in self.objects:
            if obj.name == name:
                return obj
        return None

    def get_all_objects(self) -> List[BaseObject]:
        """
        Get a list of all BaseObject instances in the scene.
        """
        return self.objects

    def initialize(self):
        self.add_object(name="plane", path=None, position=None, orientation=None)
        self.add_object(name="environment",
                        path=self.config["environment_path"],
                        position=self.config["position"],
                        orientation=self.config["orientation"])
        if self.config["objects"] is not None and self.config["objects"] is not []:
            for obj_path in self.config["object_path"]:
                self.add_object(name="object",
                                path=obj_path,
                                position=self.config["position"],
                                orientation=self.config["orientation"])
