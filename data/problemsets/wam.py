import sys
from abc import ABC

from problemset import AbstractProblemset


class Problemset(AbstractProblemset, ABC):

    @staticmethod
    def default_pose(problemset):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    @staticmethod
    def default_joint_values(problemset):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    @staticmethod
    def states(problemset):
        if problemset == "lab":
            n_states = 14
            states = [list() for _ in range(n_states)]
            states[0] = [ 0.04295548, 0.95584516, -0.96807816, 0.97116162, 0.9778903, 0.65763463, -0.68464669] # top
            states[1] = [ 0.16082985, 1.11182696, -0.92183762, 0.3794195,   1.23 ,       0.47523424, -0.27413472] # top 
            states[2] = [ 0.09952304, 1.09863569, -0.88496722, 0.38292964, 1.23, 0.41536308, -0.38031438] # top
            states[3] = [ 0.10052545, 1.06389854, -1.09858978, 0.48121717, 0.76275836, 1.38780074, 0.79727844] # top
            states[4] = [-0.45014853, 1.59318377, 0.4554682, 0.6065858, -0.38585459, 0.53452102, 0.00784768] # bottom
            states[5] = [-0.34010213,  1.6881081,   0.98402557, 0.51367941, -2.39890266, -0.58455747, 1.01213727] # bottom
            states[6] = [-0.22101804, 1.66367157, 1.09508804, 0.56299024, -2.89040372, -0.59143963, 1.31477334] # bottom
            states[7] = [-0.67729868, 1.64146044, 1.12373694, 0.91912803, -3.17152523, -0.89928808, 1.388017  ] # bottom
            states[8] = [-1.36399638, 1.91753362, 1.32779556, 2.07333031, 0.8333524, 0.08067977, -2.31735325] # bottom
            states[9] = [-0.87877812, 1.64645585, 1.34329545, 1.62880413, 0.84055928, -0.0062247, -2.29039162] # bottom
            states[10] = [ 1.38153424, 1.78324208, 0.18278696, 0.43210283, -1.62168076, 1.01491547, 2.18338891] # table
            states[11] = [ 1.60174351, 1.74358664, 0.12658995, 0.20548551, -1.48280243, 0.92108951, 2.38725579] # table
            states[12] = [ 1.9937845, 1.52197993, 0.44538624, 1.10392873, -1.28498349, 1.32703383, 2.49745328] # table
            states[13] = [-1.29228216, -1.90587936, 1.65480383, 0.20854488, 0.6896924, 0.52053023, -2.4882973 ] # table
            
        elif problemset == "industrial":
            n_states = 9
            states = [list() for _ in range(n_states)]
            states[0] = [0.9451404199026255, 0.05808521969095677, -0.3527564807017076, 1.0823735126667904, 0.08740831834786257, 0.7376408146933362, 1.5098142718954632]
            states[1] = [-1.4548690545527045, 0.40399675516342876, -0.9945734983388278, 0.37820194464231643, 1.2281017872935562, 1.009410146063119, 2.611554814425129]
            states[2] = [-1.2400057399817872, 0.5010051444244594, -0.7076935900700043, 0.25560147176881626, 1.230001164738528, 0.5877714700901333, 2.5192083678636834]
            states[3] = [0.041663608085254385, 0.34120856634159363, 0.43008822020100673, 0.702943039159143, -0.16757897004789024, 1.563316046005619, 1.9376507564619156]
            states[4] = [-0.0582735966725182, 0.3525177492274033, 0.04366314096002978, 1.0289716847892092, -1.1222444035488734, 1.5448702100587353, -1.8702188740770749]
            states[5] = [0.209152247431932, 0.39274675238945667, -0.07081202798558968, 0.83942333888157, -1.2917110514409134, 0.7491127213539474, -1.034320064356657]
            states[6] = [-0.9294721222524771, 0.3945218207628919, 0.25751742097326225, 0.635902417477271, -0.07270357592231975, 1.1093899065926003, 2.8813557378035326]
            states[7] = [0.6478256551868494, 0.4760895207119921, 0.2405648101744346, 0.6639301477277679, -0.8958582955276541, 0.7198592006957218, -0.5004250661217159]
            states[8] = [-1.7282127065673578, 0.09888218122066639, 0.19736163924951683, 1.2625983412082895, 0.2961151699367499, 0.6170075248906132, -0.034513110561670564]
            
        elif problemset == "bookshelves":
            n_states = 11
            states = [list() for _ in range(n_states)]
            states[0] = [-0.004724203785926614, 0.8920922020331432, -1.0956181628766182, 0.2888682626850513, 1.1542235723271226, 1.3258614333349312, -0.52147652663533]
            states[1] = [0.36274793480208145, 0.955484486148191, -0.6877040789915756, 0.1436648989486921, 1.1685651074027497, 0.9082915727609158, -0.8994842785311684]
            states[2] = [-0.2761372003617956, 0.8271572577090436, -0.8737703542559506, 0.4265593569916679, 1.0494484786078377, 0.9465244075502253, -0.020041897788194984]
            states[3] = [0.068548, 1.48411971, -1.44113195, 1.32035205, -0.3809645, -0.45144829, 1.95653116]
            states[4] = [0.48801217187705437, 1.3960431159005255, -1.3789132138899307, 1.8132871434460058, 0.0023114914713659846, -1.3492925819625743, 1.4114209868753094]
            states[5] = [0.5495980404686498, 1.4805171654783822, -1.3274096703807845, 0.8174673216718442, -0.22802615351116487, -1.0952276351653047, 1.6112619251726665]
            states[6] = [-0.10925527722958106, 0.45541438431124154, -0.337955026721287, 0.23303029968938907, 0.904846061599212, 0.9139043367202372, -0.38118970299238375]
            states[7] = [1.370649295965142, 1.5060056319600572, -1.211614574887084, 0.9087913853079895, 0.5874493307428117, 1.3690671586683882, -1.1275068456974742]
            states[8] = [-0.7003932265674542, 1.4883421234257534, 1.5471542434790984, 1.7664718354022626, 1.22, 1.59, -1.176334923255994]
            states[9] = [-0.97432256015594, 1.312356045557156, 2.0661115107288897, 0.1660797315277275, -1.7811733857335887, -0.33995896543955245, 0.1602472019090536]
            states[10] = [0.8814780601565136, 0.9755541639513481, 0.0902709169728008, 0.9092159375773388, -2.185473945460899, 0.4651907292730379, -2.58318530718]
        else:
            raise ValueError("Unknown problemset: {}".format(problemset))
        return n_states, states

    @staticmethod
    def joint_names(problemset):
        return [
            "wam/base_yaw_joint",
            "wam/shoulder_pitch_joint",
            "wam/shoulder_yaw_joint",
            "wam/elbow_pitch_joint",
            "wam/wrist_yaw_joint",
            "wam/wrist_pitch_joint",
            "wam/palm_yaw_joint"
            ]

    @staticmethod
    def pos_and_orn(problemset):
        if problemset == "industrial":
            return [0.0, 0.0, 0.346], [0.0, 0.0, 0.0, 1.0]
        elif problemset == "lab":
            return [0.0, 0.0, 1.3752], [0.0, 0.0, 0.0, 1.0]
        elif problemset == "bookshelves":
            return [0.0, 0.0, 0.346], [0.0, 0.0, 0.0, 1.0]
        else:
            raise ValueError("Unknown problem set: {}".format(problemset))

    @staticmethod
    def object_position(problemset):
        if problemset == "industrial":
            return [-0.2, 0.0, 0.08]
        elif problemset == "lab":
            return [0.625, 0.275, 0.85]
        elif problemset == "bookshelves":
            return [0.85, -0.15, 0.834]
        else:
            raise ValueError("Unknown problem set: {}".format(problemset))

    @staticmethod
    def planner_params(problemset):
        if problemset == "industrial":
            return {
                "sigma_obs": 0.005,
                "epsilon": 0.05,
                "lengthscales": [500.0, 500.0, 500.0, 70.0, 500.0, 500.0, 500.0],
                "variance": 0.5,
                "alpha": 100,
                "num_samples": 7,
                "num_inducing": 10,
                "learning_rate": 0.09,
                "num_steps": 130,
                "time_spacing_X": 70,
                "time_spacing_Xnew": 150
            }
        
        elif problemset == "lab":
            return {
                "sigma_obs": 0.005,
                "epsilon": 0.05,
                "lengthscales": [3] * 7,
                "variance": 0.05,
                "alpha": 100,
                "num_samples": 7,
                "num_inducing": 10,
                "learning_rate": 0.09,
                "num_steps": 130,
                "time_spacing_X": 70,
                "time_spacing_Xnew": 150
            }

        elif problemset == "bookshelves":
            return {
                "sigma_obs": 0.0005,
                "epsilon": 0.05,
                "lengthscales": [500.0, 500.0, 500.0, 200.0, 500.0, 500.0, 500.0],
                "variance": 0.5,
                "alpha": 100,
                "num_samples": 7,
                "num_inducing": 24,
                "learning_rate": 0.09,
                "num_steps": 200,
                "time_spacing_X": 100,
                "time_spacing_Xnew": 150
            }
        else:
            raise ValueError("Unknown problem set: {}".format(problemset))