import sys

from bunch import Bunch
import tensorflow as tf
from gpflow_vgpmp.utils.robot import Robot
from gpflow_vgpmp.utils.sdf_utils import SignedDensityField
from gpflow_vgpmp.utils.simulation import Simulation, ParameterLoader, Scene

# ---------------Exports
__all__ = 'simulator'


class RobotSimulator:
    def __init__(self):
        self.scene = None
        self.sdf = None
        self.robot = None
        self.plane = None
        self.sim = None
        self.config = ParameterLoader()

        self.initialize()

    def initialize(self):
        self.sim = Simulation(self.config.graphic_params)
        self.scene = Scene(self.config.scene_params)
        # self.plane = Object(name="plane")
        self.robot = Robot(self.config.robot_params)
        self.sdf = SignedDensityField.from_sdf(self.config.scene_params["sdf_path"])

    def get_simulation_params(self) -> Bunch:
        return self.config.params

    def loop(self, planner=None):
        exit = False
        while not exit:
            action = input("Enter action: ")
            if action == "q":
                exit = True
            elif action == 'c':
                print(f"Current config is :{self.robot.get_curr_config()}")
            elif action == 'sdf':
                if planner is not None:
                    self.get_rt_sdf_grad(planner)
                else:
                    print("There was no planner given")
            elif action == 'fk':
                if planner is not None:
                    joints = self.robot.get_curr_config()
                    tf.print(planner.debug_likelihood(tf.reshape(joints, (1, 1, 7))))
                else:
                    print("There was no planner given")

    def get_rt_sdf_grad(self, planner):
        """
        Get the signed distance gradient of the current robot configuration and print it
        """
        joints = self.robot.get_curr_config().reshape(7, 1)
        position = planner.likelihood.sampler._fk_cost(joints)
        print(planner.likelihood._signed_distance_grad(position))
