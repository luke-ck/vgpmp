import sys

from bunch import Bunch
import tensorflow as tf
from gpflow_vgpmp.utils.bullet_object import Object
from gpflow_vgpmp.utils.robot import Robot
from gpflow_vgpmp.utils.sdf_utils import SignedDensityField
from gpflow_vgpmp.utils.simulation import Simulation

# ---------------Exports
__all__ = 'simulator'


class RobotSimulator:
    def __init__(self):
        self.sim = Simulation()
        self.plane = Object(name="plane")
        self.robot = Robot(self.sim.robot_params)
        self.sdf = SignedDensityField.from_sdf(self.sim.scene_params["sdf_path"])
        self.scene = Object(name="scene",
                            path=self.sim.scene_params["object_path"],
                            position=self.sim.scene_params["object_position"])

    def get_simulation_params(self) -> Bunch:
        return self.sim.get_params()

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
