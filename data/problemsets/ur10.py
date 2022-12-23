import sys
from abc import ABC

from problemset import AbstractProblemset


class Problemset(AbstractProblemset, ABC):

    @staticmethod
    def default_pose(problemset):
        return [0.0, 0.5, 0.5, 0.0, 0.0, 0.0]

    @staticmethod
    def default_joint_values(problemset):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    @staticmethod
    def states(problemset):
        return 0, []

    @staticmethod
    def joint_names(problemset):
        return ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint",
                "wrist_3_joint"]
