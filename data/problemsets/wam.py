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
            states[3] = [ 0.04295548, 0.95584516, -0.96807816, 0.97116162, 0.9778903, 0.65763463, -0.68464669] # top
            states[4] = [ 0.16082985, 1.11182696, -0.92183762, 0.3794195,   1.23 ,       0.47523424, -0.27413472] # top 
            states[5] = [ 0.09952304, 1.09863569, -0.88496722, 0.38292964, 1.23, 0.41536308, -0.38031438] # top
            states[6] = [ 0.10052545, 1.06389854, -1.09858978, 0.48121717, 0.76275836, 1.38780074, 0.79727844] # top
            states[7] = [-0.45014853, 1.59318377, 0.4554682, 0.6065858, -0.38585459, 0.53452102, 0.00784768] # bottom
            states[8] = [-0.34010213,  1.6881081,   0.98402557, 0.51367941, -2.39890266, -0.58455747, 1.01213727] # bottom
            states[9] = [-0.22101804, 1.66367157, 1.09508804, 0.56299024, -2.89040372, -0.59143963, 1.31477334] # bottom
            states[10] = [-0.67729868, 1.64146044, 1.12373694, 0.91912803, -3.17152523, -0.89928808, 1.388017  ] # bottom
            states[11] = [-1.36399638, 1.91753362, 1.32779556, 2.07333031, 0.8333524, 0.08067977, -2.31735325] # bottom
            states[12] = [-0.87877812, 1.64645585, 1.34329545, 1.62880413, 0.84055928, -0.0062247, -2.29039162] # bottom
            states[13] = [ 1.38153424, 1.78324208, 0.18278696, 0.43210283, -1.62168076, 1.01491547, 2.18338891] # table
            states[14] = [ 1.60174351, 1.74358664, 0.12658995, 0.20548551, -1.48280243, 0.92108951, 2.38725579] # table
            states[15] = [ 1.9937845, 1.52197993, 0.44538624, 1.10392873, -1.28498349, 1.32703383, 2.49745328] # table
            states[16] = [-1.29228216, -1.90587936, 1.65480383, 0.20854488, 0.6896924, 0.52053023, -2.4882973 ] # table
            
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
