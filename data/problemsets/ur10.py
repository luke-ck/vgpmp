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
            states[0] = [0, 0, 0, 0, 0, 0]
            states[1] = [-1.578, -1.578, 0, 0, 0, 0]
            states[2] = [-0.35618354, -1.77651833,  0.9880922,  -0.85325163, -0.03043322, -1.77651833]
            states[3] = [ 2.42731989, -1.25568957,  0.87181485,  1.28655867, -0.02901058, 0]
        if problemset == "industrial":
            n_states = 100
            states = [list() for _ in range(n_states)]
            states[0] = [0, 0, 0, 0, 0, 0]
            states[1] = [-1.578, -1.578, 0, 0, 0, 0]
        return n_states, states

    @staticmethod
    def joint_names(problemset):
        return ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint",
                "wrist_3_joint"]
