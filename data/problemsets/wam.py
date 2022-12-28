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
        return ["wam/base_yaw_joint",
         "wam/shoulder_pitch_joint",
         "wam/shoulder_yaw_joint",
         "wam/elbow_pitch_joint",
         "wam/wrist_yaw_joint",
         "wam/wrist_pitch_joint",
         "wam/palm_yaw_joint"]
