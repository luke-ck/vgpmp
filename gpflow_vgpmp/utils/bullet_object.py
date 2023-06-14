import os
from typing import List, Optional

import pybullet as p
import pybullet_data

from .miscellaneous import suppress_stdout

__all__ = 'bullet_object'


# class Object:
#     """
#     Base class for instantiating objects in the simulator. Note that only basic operations
#     are provided, i.e. we load objects with an URDF under some orientation/position. We do this because
#     through URDF objects we create the SDF. If you want to instantiate objects without the URDF and you
#     have some other way of creating your own SDF, you will have to create a visualShape and a collisionShape
#     for the object. Details can be found in pybullet wiki. It is possible to load objects from the pybullet
#     data library.
#     """
#
#     def __init__(self, name: str = None, path: str = None, position: List = None, orientation: List = None):
#         """
#         Args:
#             name(str): name of the object being loaded. This can be used to load objects from the pybullet library.
#             Currently it is used to distinguish between the plane and everything else.
#             path(str): path to the URDF being loaded. None if loading from pybullet lib
#             position(List): len 3 array if given.
#             orientation(List): len 3 or 4 depending on whether the orientation was given in Euler angles or quaternion.
#         """
#         self.name = name
#         self.position = position
#         if orientation is not None:
#             if len(orientation) == 3:
#                 self.orientation = p.getQuaternionFromEuler(orientation)
#             elif len(orientation) == 4:
#                 self.orientation = orientation
#         else:
#             self.orientation = [0, 0, 0, 1]
#         pybullet_data_path = pybullet_data.getDataPath()
#
#         p.setAdditionalSearchPath(pybullet_data_path)  # optionally
#         object_path = None
#
#         if path is None:
#             try:
#                 if name == "plane":
#                     object_path = os.path.join(pybullet_data_path, "plane_transparent.urdf")
#                 elif name == "table":
#                     object_path = os.path.join(pybullet_data_path, "table/table.urdf")
#             except FileNotFoundError:
#                 print(f"No path for {name} found")
#                 pass
#         elif name == "scene":
#             object_path = path
#         else:
#             raise ValueError("Path for object not specified. Currently supported objects are plane and table, or"
#                              "you can specify a path to a URDF file for a scene.")
#
#         with suppress_stdout():  # suppress annoying warnings from pybullet
#             if self.position is not None:
#                 self.ID = p.loadURDF(object_path, self.position, self.orientation, useFixedBase=1,
#                                      useMaximalCoordinates=1)
#             else:
#                 self.ID = p.loadURDF(object_path)
#         print(f"Set ID to {self.ID}")
#         print(f"Created a {name}")
#
#     def set_position(self, pos):
#         self.position = pos
#
#     def get_initial_position(self):
#         return self.position
#
#     def get_current_position(self):
#         bpao = p.getBasePositionAndOrientation(self.ID)
#         return bpao[0]
#
#     def get_current_orientation(self):
#         bpao = p.getBasePositionAndOrientation(self.ID)
#         return bpao[1]
#
#     def set_orientation(self, quat):
#         self.orientation = quat
#
#     def get_initial_orientation(self):
#         return self.orientation


class BaseObject:
    """
    Base class for instantiating objects in the simulator.
    """

    def __init__(self, name: str, path: Optional[str], position: List = None, orientation: List = None):
        """
        Args:
            path(str): path to the URDF being loaded.
            position(List): len 3 array if given.
            orientation(List): len 3 or 4 depending on whether the orientation was given in Euler angles or quaternion.
        """
        self.name = name
        self.pybullet_data_path = pybullet_data.getDataPath()
        p.setAdditionalSearchPath(self.pybullet_data_path)  # optionally

        if path is None:
            # Load default scene from pybullet_data library
            object_path = os.path.join(self.pybullet_data_path, "plane_transparent.urdf")
        elif os.path.isfile(path):
            # Load scene from a path
            object_path = path
        else:
            if path == "table":
                object_path = os.path.join(self.pybullet_data_path, "table/table.urdf")
            else:
                raise ValueError("Path for object not specified. Currently supported objects are plane and table, or"
                                 "you can specify a path to a URDF file for a scene.")

        with suppress_stdout():  # suppress annoying warnings from pybullet
            if position is not None and orientation is not None:
                assert len(position) == 3, "Position must be a len 3 array"
                assert len(orientation) == 3 or len(orientation) == 4, "Orientation must be a len 3 or 4 array"
                self.position = position
                if len(orientation) == 3:
                    self.orientation = p.getQuaternionFromEuler(orientation)
                elif len(orientation) == 4:
                    self.orientation = orientation

                self.ID = p.loadURDF(object_path,
                                     self.position,
                                     self.orientation,
                                     useFixedBase=1,
                                     useMaximalCoordinates=1)
            else:
                self.ID = p.loadURDF(object_path)
        print(f"Set {self.name} ID to {self.ID}")

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

    def remove(self):
        p.removeBody(self.ID)

