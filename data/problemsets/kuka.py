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
            states[1] = [-0.8, -1.70, 1.64, 1.29, 1.1, -0.106, 2.2 - 3.1415/2 * 3]
            states[2] = [-0.24211316, 1.38721468, -0.20294616, 0.18787464, 0.0819978, -0.00455527, 0.86811348]
        if problemset == "industrial":
            n_states = 100
            states = [list() for _ in range(n_states)]
            states[0] = [0, 0, 0, 0, 0, 0, 0]

        return n_states, states

    @staticmethod
    def joint_names(problemset):
        return [
         "kuka_arm_joint_0",
         "kuka_arm_joint_1",
         "kuka_arm_joint_2",
         "kuka_arm_joint_3",
         "kuka_arm_joint_4",
         "kuka_arm_joint_5"
      ]
