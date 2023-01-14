import sys
from collections import defaultdict

import pkg_resources
import pybullet as p
import yaml
from bunch import Bunch
from pkg_resources import DistributionNotFound
from .miscellaneous import get_root_package_path

__all__ = 'simulation'


class Simulation:
    def __init__(self):
        """ 
            On init the sim loads all data from the parameter file and saves in the class as 4 different dicts:
                    - self.scene_params = scene_params["scene"]
                    - self.robot_params = robot_params["robot"]
                    - self.sim_params = sim_params["simulation"]
                    - self.graphic_params = graphic_params["graphics"]

            It also starts the simulator with or without GUI based on parameters, see parameters.

        """
        self.graphic_params = None
        self.robot_params = None
        self.sim_params = None
        self.scene_params = None
        self.physicsClient = None
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

        # input("Welcome! \nEdit the parameter file and press enter when ready...\n")

        self.load_params(self.params)
        self.start_engine()

    def load_params(self, params):
        """ Load and extract data from parameter file """
        robot_params, scene_params, trainable_params, graphic_params = params
        self.scene_params = scene_params["scene"]
        self.robot_params = robot_params["robot"]
        self.trainable_params = trainable_params["trainable_parameters"]
        self.graphic_params = graphic_params["graphics"]
        
        self.get_robot_config(self.robot_params)
        self.get_scene_config(self.scene_params)

    def get_params(self) -> Bunch:
        params = defaultdict(dict)
        params["scene"] = self.scene_params
        params["robot"] = self.robot_params
        params["trainable_params"] = self.trainable_params
        params["graphics"] = self.graphic_params
        params = Bunch(params)
        return params

    def start_engine(self):
        """ start pybullet with/without GUI """
        gravity_constant = -9.81

        if self.graphic_params["visuals"]:
            self.physicsClient = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.setRealTimeSimulation(enableRealTimeSimulation=1)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        p.resetSimulation()
        p.setGravity(0, 0, gravity_constant)

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
        config_dict = self.load_yaml_config(robot_config)
        config_dict["urdf_path"] = robot_path + config_dict["path"]

        # concatenate the robot parameters with the robot config
        # if there are any duplicates, robot parameters have precedence
        self.robot_params = {**config_dict, **robot_params}

    def get_scene_config(self, scene_params):
        """Load scene paths required for SDF and URDF files """

        problemset = scene_params["problemset"]
        object_name = scene_params["object_name"]
        sdf_name = scene_params["sdf_name"]
        scenes_data_dir_path = self.data_dir_path + "/scenes/"
        scene_path = scenes_data_dir_path + problemset + '/'
        object_path = scene_path + object_name
        sdf_path = scene_path + sdf_name
        scene_params["sdf_path"] = sdf_path
        scene_params["object_path"] = object_path

    def load_yaml_config(self, scene_config):
        with open(scene_config, 'r') as stream:
            try:
                config_dict = yaml.safe_load(stream)
            except yaml.constructor.ConstructorError as e:
                print(e)
        return config_dict
