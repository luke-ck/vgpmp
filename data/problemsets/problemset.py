import abc
from typing import List, Tuple
import sys
__all__ = "problemset"


class AbstractProblemset(object, metaclass=abc.ABCMeta):

    @staticmethod
    def active_joints(problemset: str) -> List[str]:
        raise NotImplementedError

    @staticmethod
    def default_base_pose(problemset: str) -> List[float]:
        raise NotImplementedError

    @staticmethod
    def default_joint_values(problemset: str) -> List[float]:
        raise NotImplementedError

    @staticmethod
    def problem_states(problemset: str) -> Tuple[int, List]:
        raise NotImplementedError

    @staticmethod
    def joint_names(problemset: str) -> List[str]:
        raise NotImplementedError

