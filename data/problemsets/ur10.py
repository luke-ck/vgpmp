import sys
from abc import ABC

from problemset import AbstractProblemset


class Problemset(AbstractProblemset, ABC):

    @staticmethod
    def states(problemset):
        if problemset == "bookshelves":
            n_states = 11
            states = [list() for _ in range(n_states)]
            states[0] = [-0.2249268, -1.28945924, 1.2231905, -1.77721684, -1.22327389, 0.22974946]
            states[1] = [ 0.24981678, -1.21975948,1.10768783, -1.64613056, -1.21632776, 0.22178465]
            states[2] = [-0.57713206, -1.08671742, 0.92992465, -2.06242551, -1.25622074, 0.2231967 ]
            states[3] = [-0.5353893591884503, -0.8191490171678423, 1.1005914337731868, -0.39937380083, -4.400132716711118, 1.52522899361]
            states[4] = [-0.08436517465108274, -0.9602698455366261, 1.3714961739217213, -1.80552875131, -2.1211908583364854, -0.3465566147511544]
            states[5] = [0.23993748409670962, -0.779517844077487, 1.0521555533590745, -0.24901826946, -4.450341688411703, 1.52375640995]
            states[6] = [-0.5071439050817315, -1.3731318537200201, 0.48756180486909045, 0.37845941232507224, 0.00987986784126615, 0.1334776180988649]
            states[7] = [ 0.78388833, -0.58525042, 1.07893284, -1.8885029, 1.58962479, 1.23259481]
            states[8] = [-0.21040983, -1.54239571, 2.0412341, -1.78920066, 1.53611764, 0.9651554 ]
            states[9] = [-1.0134840362055058, -0.7750392131670687, 0.5644467662671233, -0.016414804186924686, -0.0015339130707234757, 0.0007755138818968637]
            states[10] = [0.7654192050089095, -0.8439873347273738, 0.9013713739939475, -1.9100974089637022, -2.5294310029718754, 0.46645986618069685]

        elif problemset == "industrial":
            n_states = 9
            states = [list() for _ in range(n_states)]
            states[0] = [0.4013672048370986, -1.6878056769011394, 1.1177386391513888, -1.5943233518981965, -0.816752086211143, 0.08835504420274154]
            states[1] = [-1.9287061077403904, -1.4281354137637188, 0.6038045743469215, -0.9699943311452384, -1.002584041961431, -0.4831293839176354]
            states[2] = [-1.603687493717159, -1.3749979010914137, 0.624238885225218, -1.2737113215513653, -0.9836122136091396, -0.5094219490089181]
            states[3] = [-0.09618091545326236, -1.646913667854468, 1.2301884609901674, -1.3007145090883911, -0.9814376968954052, -0.5091538294726569]
            states[4] = [-0.3229351530810373, -1.4420247963969377, 0.9495182684453404, -1.469791527094406, -1.8180981506984155, -0.5806849289954321]
            states[5] = [-0.1422316204788497, -1.4244832562019099, 0.8258730902095492, -1.126235700959872, -1.7685673254265897, -0.5956613396817684]
            states[6] = [-0.9686162306332282, -1.3189068774339205, 0.8887329623521621, -1.2405571363715833, -1.677742012142283, -1.3952308744319266]
            states[7] = [0.6741154454678632, -1.1342007292487155, 0.3195164579915672, -0.7375459701998857, -1.604859458986322, -1.3194836642674819]
            states[8] = [-1.9249775378636007, -1.8529688610900832, 1.6358920957888212, -2.349581889702728, -1.0628525836094842, -0.13960430892749198]
        elif problemset == "testing":
            n_states = 2
            states = [list() for _ in range(n_states)]
            states[0] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            states[1] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            raise ValueError("Unknown problem set: {}".format(problemset))
        return n_states, states

    @staticmethod
    def pos_and_orn(problemset):
        if problemset == "industrial":
            return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]
        elif problemset == "bookshelves":
            return [0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0]
        elif problemset == "testing":
            return [0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0]
        else:
            raise ValueError("Unknown problem set: {}".format(problemset))

    @staticmethod
    def object_positions(problemset):
        if problemset == "industrial":
            return [[-0.2, 0.0, 0.08]]
        elif problemset == "bookshelves":
            return [[0.95, -0.15, 0.834]]
        elif problemset == "testing":
            return [[0.0, 0.0, 0.0]]
        else:
            raise ValueError("Unknown problem set: {}".format(problemset))

    @staticmethod
    def planner_params(problemset):
        if problemset == "industrial":
            return {
                "sigma_obs": 0.0001,
                "epsilon": 0.08,
                "lengthscales": [6, 6, 6, 6, 6, 6],
                "variance": 0.1,
                "alpha": 100,
                "num_samples": 7,
                "num_inducing": 18,
                "learning_rate": 0.02,
                "num_steps": 150,
                "time_spacing_X": 70,
                "time_spacing_Xnew": 150
            }
        elif problemset == "bookshelves":
            return {
                "sigma_obs": 0.0005,
                "epsilon": 0.05,
                "lengthscales": [4] * 6,
                "variance": 0.25,
                "alpha": 100.0,
                "num_samples": 7,
                "num_inducing": 12,
                "learning_rate": 0.02,
                "num_steps": 150,
                "time_spacing_X": 70,
                "time_spacing_Xnew": 150
            }
        elif problemset == "testing":
            return {
                "sigma_obs": 0,
                "epsilon": 0,
                "lengthscales": [0] * 6,
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
            sys.exit("Invalid problemset")