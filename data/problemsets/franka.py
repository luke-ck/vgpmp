import sys
from abc import ABC

from problemset import AbstractProblemset


class Problemset(AbstractProblemset, ABC):

    @staticmethod
    def states(problemset):
        if problemset == 'bookshelves':
            n_states = 11
            states = [list() for _ in range(n_states)]
            states[0] = [-0.31681073, 0.24296879, 0.01211371, -0.96444452, 0.43035345, 1.62566429, -0.81831454]
            states[1] = [0.37150145, 0.55337794, 0.48556757, -0.52563592, 0.02149352, 2.03900428, -1.21878495]
            states[2] = [-1.1005208, 0.61057338, 0.57576536, -0.76523535, 1.3516467, 1.5645078, -1.14941714]
            states[3] = [2.03106595, -1.43954471, -1.65219772, -0.98312804, 0.21352287, 2.57938689, -1.82386]
            states[4] = [2.78756941, -1.30129198, -1.69741135, -1.45517971, 0.7344173, 2.77076803, -1.61781752]
            states[5] = [0.97438083, -0.93748881, -1.26522131, -2.54085429, 0.2725268, 2.72170148, -1.73693474]
            states[6] = [0.4090393, -1.27601287, -0.8926530, -2.78745129, 1.53226375, 2.55162203, -1.67625164]
            states[7] = [1.68826137, -0.20236881, -2.04354627, -0.28458169, 0.5273779, 2.07144803, -0.40730323]
            states[8] = [-1.79597651, -1.76047362, -0.1279821, -0.35520999, 2.76035914, 0.89248167, -0.65049292]
            states[9] = [2.29424201, -0.84012076, 2.3159208, -0.67236157, 2.745811, 2.09272489, 0.23100074]
            states[10] = [1.35213072, 1.35108746, -1.29627257, -0.43364257, -0.66074936, 1.70439247, 0.76911109]
        elif problemset == 'industrial':
            n_states = 9
            states = [list() for _ in range(n_states)]
            states[0] = [-0.6982915, -0.47724027, 0.96112869, -1.64297994, 0.38184542, 1.28340287, 0.49280794 + 3.14159265 / 4]
            states[1] = [-1.38294485, 0.61212638, -1.31441932, -0.22121811, 1.1110808, 1.38602424, 0.81529816]
            states[2] = [-0.62041059, 0.41287353, -1.18343287, -0.71535791, 0.76938146, 1.65440323, 0.67104935]
            states[3] = [-1.80098694, 0.12521732, 1.8678778, -1.45078635, 0.02622464, 1.77610934, 2.49696111]
            states[4] = [-1.80244165, 0.22850141, 1.7566107, -1.37442941, -0.03373421, 1.39970892, 2.50909798]
            states[5] = [-1.79300457, -0.02234747, 1.95923886, -1.41112722, 0.0438824, 1.54803185, 2.50098228]
            states[6] = [-0.89457144, -0.47499645, -0.6139825, -2.16674660, -0.134227, 2.89230967, 0.00972345 + 3.14159265 / 4]
            states[7] = [2.3741933, -0.91396071, 1.510302, -0.16472317, 1.88311249, 0.95135129, 2.52099306]
            states[8] = [-2.63212708, -0.79542704, -2.61988842, -0.42029416, -0.24882192, 1.3649156, -0.52094162]
        elif problemset == 'boxes':
            n_states = 2
            states = [list() for _ in range(n_states)]
            states[0] = [-1.88327995, 0.30243233, 1.88680381, -1.32331464, 1.3319037, 1.65225616, 0.5096332]
            states[1] = [-2.88286637, -0.2759609, 0.23902162, -1.17246602, 1.19599294, 1.88570609, 0.58906756]
        elif problemset == 'testing':
            n_states = 2
            states = [list() for _ in range(n_states)]
            states[0] = [-0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0]
            states[1] = [-0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0]
        else:
            raise ValueError("Unknown problem set: {}".format(problemset))
        return n_states, states

    @staticmethod
    def pos_and_orn(problemset):
        if problemset == "industrial":
            return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]
        elif problemset == "bookshelves":
            return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]
        elif problemset == "testing":
            return [0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0]
        else:
            raise ValueError("Unknown problem set: {}".format(problemset))

    @staticmethod
    def object_positions(problemset):
        if problemset == "industrial":
            return [[-0.2, 0.0, -0.2]]
        elif problemset == "bookshelves":
            return [[0.62, -0.15, 0.834]]
        elif problemset == "boxes":
            return [[0.0, 0.0, 0.522176]]
        elif problemset == "testing":
            return [[0.0, 0.0, 0.0]]
        else:
            raise ValueError("Unknown problem set: {}".format(problemset))

    @staticmethod
    def planner_params(problemset):
        if problemset == "industrial":
            return {
                "sigma_obs": 0.005,
                "epsilon": 0.05,
                "lengthscales": [2] * 7,
                "variance": 0.2,
                "alpha": 100,
                "num_samples": 20,
                "num_inducing": 10,
                "learning_rate": 0.02,
                "num_steps": 200,
                "time_spacing_X": 50,
                "time_spacing_Xnew": 100
            }
        elif problemset == "bookshelves":
            return {
                "sigma_obs": 0.0005,
                "epsilon": 0.05,
                "lengthscales": [4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0],
                "variance": 0.5,
                "alpha": 100.0,
                "num_samples": 7,
                "num_inducing": 24,
                "learning_rate": 0.09,
                "num_steps": 130,
                "time_spacing_X": 70,
                "time_spacing_Xnew": 150
            }
        elif problemset == "boxes":
            return {
                "sigma_obs": 0.0005,
                "epsilon": 0.05,
                "lengthscales": [4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0],
                "variance": 0.5,
                "alpha": 100.0,
                "num_samples": 7,
                "num_inducing": 24,
                "learning_rate": 0.09,
                "num_steps": 130,
                "time_spacing_X": 70,
                "time_spacing_Xnew": 150
            }
        elif problemset == "testing":
            return {
                "sigma_obs": 0,
                "epsilon": 0,
                "lengthscales": [0] * 7,
                "variance": 0,
                "alpha": 0,
                "num_samples": 0,
                "num_inducing": 0,
                "learning_rate": 0,
                "num_steps": 0,
                "time_spacing_X": 0,
                "time_spacing_Xnew": 0
            }
        else:
            raise ValueError("Unknown problem set: {}".format(problemset))
