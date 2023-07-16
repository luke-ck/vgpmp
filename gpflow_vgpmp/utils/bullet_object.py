import os
import warnings
from pathlib import Path
from typing import List, Optional

import pybullet as p
import pybullet_data

from .miscellaneous import suppress_stdout

__all__ = 'bullet_object'

object_mapping = {
    "table": "table/table.urdf",
    "plane": "plane_transparent.urdf",
    "cube": "cube.urdf",
    "duck": "duck_vhacd.urdf"
}


class BaseObject:
    """
    Base class for instantiating objects in the simulator.
    """

    def __init__(self, name: str, path: Optional[Path], position: List = None, orientation: List = None,
                 client: int = 0):
        """
        Either provide the name of the object or the path to the URDF file. If both are provided, the path has priority.
        Args:
            name(str): name of the object. If path is not given, this is used to load the object from pybullet data.
            path(pathlib.Path): path to the URDF being loaded.
            position(List): len 3 array if given.
            orientation(List): len 3 or 4 depending on whether the orientation was given in Euler angles or quaternion.
            client(int): pybullet client ID.
        """

        pybullet_data_path = pybullet_data.getDataPath()
        assert p.isConnected(client), "Pybullet client not connected"
        p.setAdditionalSearchPath(pybullet_data_path)  # optionally
        pybullet_data_path = Path(pybullet_data_path)

        if path is not None:
            assert path.is_file() and path.exists(), "Path does not exist"
            assert path.suffix == ".urdf", "Only URDF files are supported"

            object_name = path.stem
            if object_name != name:
                warnings.warn(f"Name of the object {name} does not match the name of of object from path {object_name}. "
                              f"Using the name from the path.")

            object_path = path
            name = object_name
        else:
            try:
                object_path = pybullet_data_path / object_mapping[name]
                assert object_path.is_file() and object_path.exists(), "Path does not exist"
            except KeyError:
                raise KeyError(f"Object {name} not found in the object_mapping dictionary of supported objects. If "
                               f"you want to load a custom object, please provide the path to the URDF file or modify "
                               f"the dictionary.")

        # with suppress_stdout():  # this breaks tests. suppress annoying warnings from pybullet.
        if position is not None and orientation is not None:
            assert len(position) == 3, "Position must be a len 3 array"
            assert len(orientation) == 3 or len(orientation) == 4, "Orientation must be a len 3 or 4 array"

            if len(orientation) == 3:
                orientation = p.getQuaternionFromEuler(orientation)
        else:
            position = [0, 0, 0]
            orientation = [0, 0, 0, 1]
        self.orientation = orientation
        self.position = position
        self.name = name
        self.ID = p.loadURDF(object_path.as_posix(),
                             self.position,
                             self.orientation,
                             useFixedBase=1,
                             useMaximalCoordinates=1)
        self.path = object_path
        print(f"Set {self.name} ID to {self.ID}")

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, pos):
        self._position = pos

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, quat):
        self._orientation = quat

    def get_current_position(self):
        bpao = p.getBasePositionAndOrientation(self.ID)
        return bpao[0]

    def get_current_orientation(self):
        bpao = p.getBasePositionAndOrientation(self.ID)
        return bpao[1]

    def set_current_position(self):
        pos = self.get_current_position()
        self.position = pos

    def set_current_orientation(self):
        orn = self.get_current_orientation()
        self.orientation = orn

    def remove(self):
        p.removeBody(self.ID)
