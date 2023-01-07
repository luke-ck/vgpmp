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
        # self.table = Object(name="table", position=[-0.65, 0.9, 0])
        # self.scene = Object(name="scene",
        #                     path=self.sim.scene_params["object_path"],
        #                     position=[100, 100, 100])

    def get_simulation_params(self) -> Bunch:
        return self.sim.get_params()

    def loop(self, planner=None):
        while True:
            if input() == "q":
                break
            elif input() == 'a':
                print(self.robot.get_curr_config())

            elif input() == 'v':
                if planner is not None:
                    joints = self.robot.get_curr_config()
                    position = planner.likelihood.sampler._fk_cost(joints.reshape(7, 1))
                    print(planner.likelihood._signed_distance_grad(position))

                else:
                    print("There was no planner given")
            elif input() == 'c':
                if planner is not None:
                    joints = self.robot.get_curr_config()
                    tf.print(planner.debug_likelihood(tf.reshape(joints, (1, 1, 7))))

                else:
                    print("There was no planner given")

    def get_rt_sdf(self, planner):
        joints = self.robot.get_curr_config()
        position = planner.likelihood.sampler._fk_cost(joints.reshape(7, 1))
        print(planner.likelihood._signed_distance_grad(position))
