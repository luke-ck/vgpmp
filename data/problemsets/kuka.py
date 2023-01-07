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
            states[3] = [0.2853338, -0.31229435, 0.66712463, 1.27568255, 0.77873262, -2.00490558, 0.03039512]
            states[4] = [0.98900646, -0.52004561, 0.65509585, 1.27189863, 0.74729805, -1.99651917, 0.0345732]
            states[5] = [-1.20984506, -0.23616358, -0.32719039, 0.57762214, -0.0099703, -1.40514764, -0.07853945]
            states[6] = [-2.79421256, -0.45425777, -0.02895163, 1.14678609, 0.13632731, 0.76368015, -0.66330268]
            states[7] = [ 2.05141915, -0.80958829, 1.00458258, 0.99531097, -0.20972174, 0.02712899, 0.10857331]
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
