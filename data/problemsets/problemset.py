import abc
from typing import List, Tuple

__all__ = "problemset"


class AbstractProblemset(object, metaclass=abc.ABCMeta):

    @staticmethod
    def default_pose(problemset: str) -> List[float]:
        """
        Returns the default pose for the given problemset environment.
        The base pose refers to the entire robot, not just the DOF.
        """
        raise NotImplementedError

    @staticmethod
    def problem_states(problemset: str) -> Tuple[int, List]:
        raise NotImplementedError

    @staticmethod
    def joint_names(problemset: str) -> List[str]:
        """
        The joint names for all joints in the robot
        """
        raise NotImplementedError

