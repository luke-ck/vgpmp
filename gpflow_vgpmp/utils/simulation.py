import sys
from collections import defaultdict

import pkg_resources
import pybullet as p
import yaml
from bunch import Bunch
from pkg_resources import DistributionNotFound

__all__ = 'simulation'


class Simulation:
    def __init__(self):
        """ 
            On init the sim loads all data from the parameter file and saves in the class as 5 different dicts:
                    - self.scene_params = scene_params["scene"]
                    - self.robot_params = robot_params["robot"]
                    - self.planner_params = planner_params["planner"]
                    - self.sim_params = sim_params["simulation"]
                    - self.graphic_params = graphic_params["graphics"]

            It also starts the simulator with or without GUI based on parameters, see parameters.

        """
        self.planner_params = None
        self.graphic_params = None
        self.robot_params = None
        self.sim_params = None
        self.scene_params = None
        self.physicsClient = None

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
        robot_params, scene_params, planner_params, trainable_params, sim_params, graphic_params = params
        self.scene_params = scene_params["scene"]
        self.robot_params = robot_params["robot"]
        self.planner_params = planner_params["planner"]
        self.trainable_params = trainable_params["trainable_parameters"]
        self.sim_params = sim_params["simulation"]
        self.graphic_params = graphic_params["graphics"]

    def get_params(self) -> Bunch:
        params = defaultdict(dict)
        params["scene"] = self.scene_params
        params["robot"] = self.robot_params
        params["planner"] = self.planner_params
        params["trainable_params"] = self.trainable_params
        params["simulation"] = self.sim_params
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
