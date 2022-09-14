import sys

from bunch import Bunch

from gpflow_vgpmp.utils.bullet_object import Object
from gpflow_vgpmp.utils.robot import Robot
from gpflow_vgpmp.utils.sdf_utils import SignedDensityField
from gpflow_vgpmp.utils.simulation import Simulation

# ---------------Exports
__all__ = ('simulator')


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

    def loop(self):
        while True:
            if input() == "exit":
                sys.exit()
