import os
from typing import List

import pybullet as p
import pybullet_data

from .miscellaneous import suppress_stdout

meshScale = [1.5, 1.5, 1.5]
ObjIniOrientation = p.getQuaternionFromEuler([0, 0, 0])
basePos = [0, 0, 0]
inertial = [0, 0, 0]
shift = [0, 0, 0]

__all__ = 'bullet_object'


class Object:
    """
    Base class for instantiating objects in the simulator. Note that only basic operations
    are provided, i.e. we load objects with an URDF under some orientation/position. We do this because
    through URDF objects we create the SDF. If you want to instantiate objects without the URDF and you
    have some other way of creating your own SDF, you will have to create a visualShape and a collisionShape
    for the object. Details can be found in pybullet wiki. It is possible to load objects from the pybullet
    data library.
    """

    def __init__(self, name: str = None, path: str = None, position: List = None, orientation: List = None):
        """
        Args:
            name(str): name of the object being loaded. This can be used to load objects from the pybullet library.
            Currently it is used to distinguish between the plane and everything else.
            path(str): path to the URDF being loaded. None if loading from pybullet lib
            position(List): len 3 array if given.
            orientation(List): len 3 or 4 depending on whether the orientation was given in Euler angles or quaternion.
        """
        self.name = name
        self.position = position
        if orientation is not None:
            if len(orientation) == 3:
                self.orientation = p.getQuaternionFromEuler(orientation)
            elif len(orientation) == 4:
                self.orientation = orientation
        else:
            self.orientation = [0, 0, 0, 1]
        pybullet_data_path = pybullet_data.getDataPath()

        p.setAdditionalSearchPath(pybullet_data_path)  # optionally
        object_path = None

        if path is None:
            try:
                if name == "plane":
                    object_path = os.path.join(pybullet_data_path, "plane_transparent.urdf")
                elif name == "table":
                    object_path = os.path.join(pybullet_data_path, "table/table.urdf")
            except FileNotFoundError:
                print(f"No path for {name} found")
                pass
        elif name == "scene":
            object_path = path
        elif name == "pringles":
            object_path = path
        else:
            raise ValueError("Path for object not specified. Currently supported objects are plane and table, or"
                             "you can specify a path to a URDF file for a scene.")

        with suppress_stdout(): # suppress annoying warnings from pybullet
            if self.position is not None:
                self.ID = p.loadURDF(object_path, self.position, self.orientation)
            else:
                self.ID = p.loadURDF(object_path)
        print(f"Set ID to {self.ID}")
        print(f"Created a {name}")

    def set_position(self, pos):
        self.position = pos

    def get_initial_position(self):
        return self.position

    def get_current_position(self):
        bpao = p.getBasePositionAndOrientation(self.ID)
        return bpao[0]

    def get_current_orientation(self):
        bpao = p.getBasePositionAndOrientation(self.ID)
        return bpao[1]

    def set_orientation(self, quat):
        self.orientation = quat

    def get_initial_orientation(self):
        return self.orientation
