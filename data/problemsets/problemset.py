import abc
import importlib
import itertools
import os
import sys
from data.problemsets import config
from typing import List, Tuple

__all__ = "problemset"


def import_problemset(robot_name):

    if robot_name not in config.robot_modules:
        print("Robot not available. Check params file and try again... The simulator will now exit.")
        sys.exit(-1)
    module_name = config.robot_modules[robot_name]
    module = importlib.import_module(module_name)
    return module.Problemset


def create_problems(problemset_name: str, robot_name: str) -> Tuple[List[Tuple[List, List]], dict, List[str], List[float], List[float]]:
    r"""
    For the given problemset and robot names, returns the combination of all possible problems,
    the planner parameters for the given environment and robot, and the
    robot joint names, their default pose and the robot position in world coordinates.
    """
    # Start and end joint angles
    Problemset = import_problemset(robot_name)
    n_states, states = Problemset.states(problemset_name)
    print('There are %s total robot positions' % n_states)
    # all possible combinations of 2 pairs
    benchmark = list(itertools.combinations(states, 2))
    print('And a total of %d problems in the %s problemset' %
          (len(benchmark), problemset_name))
    robot_joint_names = Problemset.joint_names(problemset_name)
    robot_default_pose = Problemset.default_pose(problemset_name)
    planner_params = Problemset.planner_params(problemset_name)
    robot_pos_and_orn = Problemset.pos_and_orn(problemset_name)

    return benchmark, planner_params, robot_joint_names, robot_default_pose, robot_pos_and_orn


class AbstractProblemset(object, metaclass=abc.ABCMeta):

    @staticmethod
    def default_pose(problemset: str) -> List[float]:
        """
        Returns the default pose for the given problemset environment.
        The base pose refers to the entire robot, not just the DOF.
        """
        raise NotImplementedError

    @staticmethod
    def states(problemset: str) -> Tuple[int, List]:
        raise NotImplementedError

    @staticmethod
    def joint_names(problemset: str) -> List[str]:
        """
        The joint names for all joints in the robot
        """
        raise NotImplementedError

    @staticmethod
    def pos_and_orn(problemset: str) -> Tuple[List[float], List[float]]:
        """
        The position and orientation of the robot in the world frame
        """
        raise NotImplementedError

    @staticmethod
    def planner_params(problemset: str) -> dict:
        """
        The parameters for the planner
        """
        raise NotImplementedError

