import itertools
import sys
from typing import Tuple, List

import numpy as np
from bunch import Bunch
import tensorflow as tf

from data.problemsets.problemset import import_problemset
from gpflow_vgpmp.utils.robot import Robot
from gpflow_vgpmp.utils.sdf_utils import SignedDensityField
from gpflow_vgpmp.utils.simulation import Simulation, Scene
from gpflow_vgpmp.utils.parameter_loader import ParameterLoader

# ---------------Exports
__all__ = 'simulator'


class SimulationManager:
    def __init__(self, parameter_file_path=None):
        self.scene = None
        self.sdf = None
        self.robot = None
        self.sim = None
        self._config = ParameterLoader(parameter_file_path)

        self.initialize()

    @property
    def config(self) -> dict:
        return self._config.params

    def initialize(self):
        self.sim = Simulation(self.config['graphic_params'])
        self.scene = Scene(self.config['scene_params'])
        self.robot = Robot(self.config['robot_params'])
        self.sdf = SignedDensityField.from_sdf(self.config['scene_params']["sdf_path"])

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
