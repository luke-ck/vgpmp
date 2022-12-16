import sys
from abc import ABC

from problemset import AbstractProblemset


class Problemset(AbstractProblemset, ABC):

    @staticmethod
    def active_joints(problemset):
        if problemset == 'bookshelves':
            return ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
                    'panda_joint5', 'panda_joint6', 'panda_joint7']
        if problemset == 'industrial':
            return ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
                    'panda_joint5', 'panda_joint6', 'panda_joint7']
        else:
            print("Unknown problem set")
            sys.exit()

    @staticmethod
    def default_base_pose(problemset):
        if problemset == 'bookshelves':
            return [0] * 7
        if problemset == 'industrial':
            return [0] * 7
        else:
            print("Unknown problem set")
            sys.exit()

    @staticmethod
    def default_joint_values(problemset):
        if problemset == 'bookshelves':
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if problemset == 'industrial':
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            print("Unknown problem set")
            sys.exit()

    @staticmethod
    def joint_names(problemset):
        if problemset == 'bookshelves':
            return ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
                    'panda_joint5', 'panda_joint6', 'panda_joint7', 'panda_joint8',
                    'panda_hand_joint', 'panda_finger_joint1', 'panda_finger_joint2']
        if problemset == 'industrial':
            return ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
                    'panda_joint5', 'panda_joint6', 'panda_joint7', 'panda_joint8',
                    'panda_hand_joint', 'panda_finger_joint1', 'panda_finger_joint2']
        else:
            print("Unknown problem set")
            sys.exit()

    @staticmethod
    def states(problemset):
        if problemset == 'bookshelves':
            n_states = 19
            states = [list() for _ in range(n_states)]
            states[0] = [-0.5, 0.1, 0.45, -1.27, 0.0, 0.0, 0.5]
            states[1] = [0.5, 0.156, 0.225, -0.88, 0.0, 0.0, 0.5]
            states[2] = [2.89, -0.11574178, 2.89, -1.22103215, 0.64931876, 2.05112939, 1.34036701]
            states[3] = [2.89, -0.07728433, 2.85034839, -0.11602217, 0.90698432, 1.58977069, -0.22082354]
            states[4] = [-1.58059445, -1.65310661, -0.14810233, -0.30461756, -2.44013792, 1.14956004, 1.95201477]
            states[5] = [2.70168464, -0.21771183, 2.89, -2.17705658, 1.94299994, 1.25105265, 1.30432147]
            states[6] = [1.42879417, 0.69999396, -0.30443448, -0.51113233, 1.84498211, 3.15694314, 2.80417896]
            states[7] = [1.99846448, -1.26152362, 2.57726846, -0.29399722, 2.10806173, 2.32986863, 2.73081491]
            states[8] = [1.10337441, 0.30618956, 0.82820592, -1.1688561, 2.14047465, 3.32691917, -2.52947092]
            states[9] = [0.94834949, 0.27070891, 0.9458488, -1.20790021, 2.87953749, 3.30338553, -2.89]
            states[10] = [0.95389691, 0.28747582, 0.84152873, -1.12633372, 2.78919892, 3.12999691, -2.89]
            states[11] = [0.83155074, 0.66376191, -2.20639318, -1.28248429, 0.98033752, 1.00804679, -0.94676979]
            states[12] = [1.23358922, 0.6803201, -2.14249568, -1.10971056, 0.89706398, 0.90018203, -1.13474725]
            states[13] = [1.01442439, 0.05880181, -0.89710597, -1.4693433, 1.54635831, 1.33902106, -1.36964488]
            states[14] = [2.89666603, -1.51944122, -1.2168953, -0.84851096, 2.89691345, 2.18001039, -1.99308977]
            states[15] = [2.68410936, -1.53257679, -1.26748437, -0.6310793, 2.86421964, 2.22596736, -1.99310638]
            states[16] = [0.89994029, 0.53980228, -2.897104, -0.1966807, -2.76516711, 2.14448415, -0.2313831]
            states[17] = [-0.93438354, 0.70995129, -2.72378345, -0.43366561, -2.79207336, 1.97666147, -0.423157]
            states[18] = [ 0.45692096, 0.66025557, -1.73086682, -1.05139134, -0.22906974, 2.15095555, -0.95843739]
            return n_states, states
        if problemset == 'industrial':
            n_states = 12
            states = [list() for _ in range(n_states)]
            states[0] = [-0.5, 0.1, 0.45, -1.27, 0.0, 0.0, 0.5]
            states[1] = [0.5, 0.156, 0.225, -0.88, 0.0, 0.0, 0.5]
            states[2] = [-1.38294485, 0.61212638, -1.31441932, -0.22121811, 1.1110808, 1.38602424, 0.81529816]
            states[3] = [-0.62041059, 0.41287353, -1.18343287, -0.71535791, 0.76938146, 1.65440323, 0.67104935]
            states[4] = [-1.80098694, 0.12521732, 1.8678778, -1.45078635, 0.02622464, 1.77610934, 2.49696111]
            states[5] = [-1.80244165, 0.22850141, 1.7566107, -1.37442941, -0.03373421, 1.39970892, 2.50909798]
            states[6] = [-1.79300457, -0.02234747, 1.95923886, -1.41112722, 0.0438824, 1.54803185, 2.50098228]
            states[7] = [-0.51877685, 0.38124115, 0.7164529, -1.1444525, -0.15082004, 1.8269117, 2.8963512]
            states[8] = [2.3741933, -0.91396071, 1.510302, -0.16472317, 1.88311249, 0.95135129, 2.52099306]
            states[9] = [-0.23894247, 0.43614839, -0.03961347, -0.19530461, 0.44537682, 2.07451343, -0.30374485]
            states[10] = [-1.7410423, -0.29370995, 0.15012496, -2.12489589, 0.63558159, 2.72581121, 2.72380461]
            states[11] = [ 2.6476631, 0.00832994, -2.67830239, -1.43222436, 2.80934852, 3.25453773, 2.89]
            return n_states, states

        else:
            print("Unknown problem set")
            sys.exit()
