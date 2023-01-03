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
        if problemset == "bookshelves":
            n_states = 100
            states = [list() for _ in range(n_states)]
            states[0] = [0, 0, 0, 0, 0, 0, 0]
            states[1] = [-0.8, -1.70, 1.64, 1.29, 1.1, -0.106, 2.2]
            states[2] = [-0.24211316, 1.38721468, -0.20294616, 0.18787464, 0.0819978, -0.00455527, 0.86811348]
        if problemset == "industrial":
            n_states = 100
            states = [list() for _ in range(n_states)]
            states[0] = [0, 0, 0, 0, 0, 0, 0]

        return n_states, states

    @staticmethod
    def joint_names(problemset):
        return ["wam/base_yaw_joint",
         "wam/shoulder_pitch_joint",
         "wam/shoulder_yaw_joint",
         "wam/elbow_pitch_joint",
         "wam/wrist_yaw_joint",
         "wam/wrist_pitch_joint",
         "wam/palm_yaw_joint"]
