import sys
from abc import ABC

from problemset import AbstractProblemset


class Problemset(AbstractProblemset, ABC):

    @staticmethod
    def default_pose(problemset):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    @staticmethod
    def states(problemset):
        if problemset == "bookshelves":
            n_states = 11
            states = [list() for _ in range(n_states)]
            states[0] = [-2.70241973, -0.49613738, -1.21524911, 1.19175442, -1.66793785, 1.4216143, 1.40841643]
            states[1] = [-2.60980499, -0.55524293, 0.05467749, 0.71731258, -2.50983548, 1.35129081, 1.22302333]
            states[2] = [-1.04673182, 0.68041032, 0.63478252, -0.64227376, -2.95, -1.07312957, 0.96852192]
            states[3] = [-0.14562691, 1.55074569, 1.64963291, 1.12700839, -2.96358063, -0.96531264, 2.71841159]
            states[4] = [0.95010236, 0.9412263, 1.86583927, 2.06707834, -0.12104962, -0.41134678, 2.38845786]
            states[5] = [0.7185098, 1.5467082, 1.59868419, 0.48303008, -2.95, -0.89603846, 2.68993638]
            states[6] = [-2.795486, -0.33384941, 1.23002965, -0.12650341, 1.83863193, 1.33100174, 1.76342373]
            states[7] = [-2.91136155, -0.29422539, -0.7016362, 0.91150894, 0.71475798, -0.11987578, 1.58946973]
            states[8] = [-1.63475427, -1.16351356, -0.2439754, 1.67138724, -0.676114, 0.09897403, -0.42969901]
            states[9] = [-1.65675007, -1.56161412, 1.41763991, -0.735414, -0.05649168, 0.83173543, -0.36160198]
            states[10] = [1.90321228, -1.3967333, 2.1108019, 0.17666078, -0.6683322, -0.99942099, -0.78571672]
        elif problemset == "industrial":
            n_states = 9
            states = [list() for _ in range(n_states)]
            states[0] = [-0.40559687, 0.90881157, 0.67154698, -0.42949893, 1.38729146, 0.08476077, 0.46148802]
            states[1] = [0.46677742, 0.90725565, -0.71711314, -0.65490859, -1.26788683, -0.17395347, 0.36174169]
            states[2] = [-1.56802787, -0.92442251, 1.24349469, -1.40102403, 2.16888415, 2.03915208, 1.2007262]
            states[3] = [-2.21041063, -0.89597717, 1.80050713, -0.51560786, 2.91583137, -0.44374258, 1.88469752]
            states[4] = [-0.94672464, 0.60709815, -2.94705973, 0.96348489, -2.63712788, 0.82603669, 0.50394097]
            states[5] = [-1.78810809, 0.17870216, -2.95667161, 1.6378849, -1.01828982, 0.0606072, 0.65012045]
            states[6] = [-1.35057625, 0.5838822, -2.77513306, 0.51041783, -0.65820477, -0.82270028, 0.35602494]
            states[7] = [-1.65521238, 0.75530852, -2.8386648, 0.14505213, 0.00402573, -1.03228289, 0.29148176]
            states[8] = [1.2179725, 1.19623672, 0.73636572, 1.46116258, 1.54880835, -1.90864908, 2.67156503]
        elif problemset == "boxes":
            n_states = 2
            states = [list() for _ in range(n_states)]
            states[0] = [-1.88327995, 0.30243233, 1.88680381, -1.32331464, 1.3319037, 1.65225616, 0.5096332]
            states[1] = [-2.88286637, -0.2759609, 0.23902162, -1.17246602, 1.19599294, 1.88570609, 0.58906756]
        else:
            raise ValueError("Unknown problem set: {}".format(problemset))
        return n_states, states

    @staticmethod
    def joint_names(problemset):
        return [
            "iiwa_description_joint_1",
            "iiwa_description_joint_2",
            "iiwa_description_joint_3",
            "iiwa_description_joint_4",
            "iiwa_description_joint_5",
            "iiwa_description_joint_6",
            "iiwa_description_joint_7",
            "iiwa_description_joint_ee"
        ]

    @staticmethod
    def pos_and_orn(problemset):
        if problemset == "industrial":
            return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]
        elif problemset == "bookshelves":
            return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]
        else:
            raise ValueError("Unknown problem set: {}".format(problemset))

    @staticmethod
    def object_position(problemset):
        if problemset == "industrial":
            return [-0.2, 0.0, -0.2]
        elif problemset == "bookshelves":
            return [0.62, -0.15, 0.834]
        else:
            raise ValueError("Unknown problem set: {}".format(problemset))
            sys.exit("Invalid problemset")

    @staticmethod
    def planner_params(problemset):
        if problemset == "industrial":
            return {
                "sigma_obs": 0.001,
                "epsilon": 0.05,
                "lengthscales": [2] * 7,
                "variance": 0.3,
                "alpha": 100,
                "num_samples": 20,
                "num_inducing": 7,
                "learning_rate": 0.02,
                "num_steps": 200,
                "time_spacing_X": 50,
                "time_spacing_Xnew": 100
            }
        elif problemset == "bookshelves":
            return {
                "sigma_obs": 0.0001,
                "epsilon": 0.05,
                "lengthscales": [3] * 7,
                "variance": 0.3,
                "alpha": 100,
                "num_samples": 20,
                "num_inducing": 10,
                "learning_rate": 0.02,
                "num_steps": 200,
                "time_spacing_X": 100,
                "time_spacing_Xnew": 150
            }
        elif problemset == "boxes":
            return Problemset.planner_params("bookshelves")
        else:
            raise ValueError("Unknown problem set: {}".format(problemset))
