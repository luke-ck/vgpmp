import abc
import importlib
import itertools
import os
import sys
from data.problemsets import config
from typing import List, Tuple

__all__ = "problemset"

from gpflow_vgpmp.utils.miscellaneous import get_root_package_path


def import_problemset(robot_name):
    problemset_path = get_root_package_path() + '/data/problemsets/'
    if robot_name not in config.robot_modules:
        print("Robot not available. Check params file and try again... The simulator will now exit.")
        sys.exit(-1)
    module_name = config.robot_modules[robot_name]
    sys.path.append(problemset_path)
    module = importlib.import_module(module_name)
    return module.Problemset


class AbstractProblemset(object, metaclass=abc.ABCMeta):

    @staticmethod
    def states(problemset: str) -> Tuple[int, List]:
        raise NotImplementedError

    @staticmethod
    def object_positions(problemset: str) -> List[List[float]]:
        """
        The positions of the objects in the world frame
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

