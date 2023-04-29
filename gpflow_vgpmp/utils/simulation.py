import sys
import threading
from collections import defaultdict
from typing import List, Optional, Tuple

import pkg_resources
import pybullet as p
import yaml
from bunch import Bunch
from pkg_resources import DistributionNotFound

from .bullet_object import BaseObject
from .miscellaneous import get_root_package_path

__all__ = 'simulation'


def load_yaml_config(scene_config):
    with open(scene_config, 'r') as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.constructor.ConstructorError as e:
            print(e)
    return config_dict


class ParameterLoader:
    """
       Load and extract data from a parameter files, which are used to configure the simulator.
       On start up, the loader configures the following parameters:
               - self.scene_params = scene_params["scene"]
               - self.robot_params = robot_params["robot"]
               - self.sim_params = sim_params["simulation"]
               - self.graphic_params = graphic_params["graphics"]
   """

    def __init__(self):
        self.trainable_params = None
        self.graphic_params = None
        self.robot_params = None
        self.scene_params = None
        self.data_dir_path = get_root_package_path() + "/data/"

        try:
            with open('requirements.txt', 'r') as req:
                pkg_resources.require(req)
        except DistributionNotFound:
            print('[Warning]: Missing packages ')
            sys.exit(
                '[EXIT]: System will exit, please install all requirements and run the simulator again')

        try:
            stream = open("parameters.yaml", 'r')
        except IOError:
            print("[Error]: parameters file could not be found ")
            sys.exit('[EXIT]: System will exit, please provide a parameter file and try again')

        try:
            self.params = yaml.safe_load(stream)
        except yaml.constructor.ConstructorError as e:
            print(e)

        self.load_params(self.params)

    def load_params(self, params):
        """ Load and extract data from parameter file """
        robot_params, scene_params, trainable_params, graphic_params = params
        self.scene_params = scene_params["scene"]
        self.robot_params = robot_params["robot"]
        self.trainable_params = trainable_params["trainable_parameters"]
        self.graphic_params = graphic_params["graphics"]

        self.get_robot_config(self.robot_params)
        self.get_scene_config(self.scene_params)

        self.params = Bunch(
            scene_params=self.scene_params,
            robot_params=self.robot_params,
            trainable_params=self.trainable_params,
            graphic_params=self.graphic_params
        )

    def get_robot_config(self, robot_params):
        """ Load robot configuration from parameter file which is found together with the rest of the robot files
            The robot configuration is saved in the robot_params dict.
            The general idea is this: general robot parameters are saved in the main parameter file,
            and robot specific parameters (FK related, joint limits, spheres and so on)
            are saved in the robot specific parameter file.
         """
        robot_name = robot_params["robot_name"]  # specify which robot to load
        robots_data_dir_path = self.data_dir_path + "/robots/"  # TODO: make data dir path a parameter?
        robot_path = robots_data_dir_path + robot_name + "/"
        robot_config = robot_path + "config.yaml"
        config_dict = load_yaml_config(robot_config)
        config_dict["urdf_path"] = robot_path + config_dict["path"]

        # concatenate the robot parameters with the robot config
        # if there are any duplicates, robot parameters have precedence
        self.robot_params = {**config_dict, **robot_params}

    def get_scene_config(self, scene_params):
        """Load scene paths required for SDF and URDF files """

        problemset = scene_params["problemset"]
        environment_name = scene_params["environment_name"]
        sdf_name = scene_params["sdf_name"]

        scene_params["object_path"] = []
        if scene_params["objects"] is not None and scene_params["objects"] is not []:
            objects_data_dir_path = self.data_dir_path + "/objects/"
            for obj in scene_params["objects"]:
                object_path = objects_data_dir_path + obj
                scene_params["object_path"].append(object_path)

        environment_path, sdf_path = self.get_assets_path(problemset, environment_name, sdf_name)

        scene_params["sdf_path"] = sdf_path
        scene_params["environment_path"] = environment_path

        self.scene_params = scene_params

    def get_assets_path(self, problemset: str, environment_name: str, sdf_name: str) -> Tuple[str, str]:
        scenes_data_dir_path = self.data_dir_path + "/scenes/"
        scene_path = scenes_data_dir_path + problemset + '/' + environment_name + '.urdf'
        sdf_path = scenes_data_dir_path + problemset + '/' + sdf_name + '.sdf'
        return scene_path, sdf_path


class Simulation:
    def __init__(self, graphic_params):
        self.scene = None
        self.client = None
        self.graphic_params = graphic_params

        self.start_engine()

    def start_engine(self):
        """ start pybullet with/without GUI """
        gravity_constant = -9.81

        if self.graphic_params["visuals"]:
            self.client = p.connect(p.GUI, options="--width=960 "
                                                   "--height=540 ")
            # "--mp4=\"video.mp4\"")
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.setRealTimeSimulation(enableRealTimeSimulation=1)
            # Start the window monitor thread
            # monitor_thread = threading.Thread(target=window_monitor)
            # monitor_thread.start()
        else:
            self.client = p.connect(p.DIRECT)

        p.resetSimulation()
        p.setGravity(0, 0, gravity_constant)


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
            for idx, obj_path in enumerate(self.config["object_path"]):
                self.add_object(name="object",
                                path=obj_path,
                                position=self.config["position"],
                                orientation=self.config["orientation"])
