from collections import deque
from pathlib import Path
from typing import Optional, List

from gpflow_vgpmp.utils.bullet_object import BaseObject
from gpflow_vgpmp.utils.parameter_loader import ParameterLoader


class Scene:
    """
    Class for managing a scene of BaseObjects.
    """

    def __init__(self, config: ParameterLoader = None, client: int = None):
        """
        Create a new Scene instance.
        """
        self.orientation = None
        self.position = None
        self.is_initialized = None
        try:
            self.config = config.scene_params
        except AttributeError:
            self.config = None
        self.client = client
        # FIFO queue for storing the objects
        self.objects = deque()

    def initialize(self, client):
        self.client = client
        assert self.config is not None, "Scene config is not initialized"
        self.add_object(name="plane", path=None, position=None, orientation=None)
        self.add_object(name="environment",
                        path=self.config["environment_path"],
                        position=self.config["position"],
                        orientation=self.config["orientation"])
        if self.config["objects"] is not None and self.config["objects"] is not []:
            for index, obj_path in enumerate(self.config["objects_path"]):
                obj_position = self.config["objects_position"][index]
                obj_orientation = self.config["objects_orientation"][index]
                obj_name = self.config['objects_path'][index].stem  # get the name of the file without the extension
                self.add_object(name=obj_name,
                                path=obj_path,
                                position=obj_position,
                                orientation=obj_orientation)

        self.position = self.config["position"]
        self.orientation = self.config["orientation"]
        self.is_initialized = True

    def add_object(self, name: str, path: Optional[str], position: Optional[List], orientation: Optional[List]):
        """
        Add a new BaseObject instance to the scene.
        """
        path = Path(path) if type(path) == str else path  # Convert path to Path object if it is a string
        assert self.client is not None, "No client is connected to the backend"
        obj = BaseObject(name=name, path=path, position=position, orientation=orientation, client=self.client)
        self.objects.appendleft(obj)

    def remove_object(self, obj: Optional[BaseObject] = None, index: Optional[int] = None):
        """
        Remove a BaseObject instance from the scene.
        """
        assert obj is not None or index is not None, "Either obj or index must be specified"
        obj.remove()
        if index is not None:
            self.objects.remove(self.objects[index])
        else:
            # remove last occurence of obj from list
            for item in self.objects:
                if item == obj:
                    self.objects.remove(item)
                    return
            raise ValueError("Object not found in scene")

    def remove_object_by_name(self, name: str):
        """
        Remove a BaseObject instance from the scene by name.
        """
        obj = self.get_object_by_name(name)
        if obj is not None:
            self.remove_object(obj)
        else:
            raise ValueError("Object not found in scene")

    def get_object_by_name(self, name: str) -> Optional[BaseObject]:
        """
        Get a BaseObject instance from the scene by name.

        Args:
            name(str): The name of the BaseObject instance to retrieve.

        Returns:
            The BaseObject instance with the specified name, or None if not found.
        """
        for obj in self.objects:
            if obj.name == name and type(obj) == BaseObject:
                return obj
        return None

    def get_object_by_index(self, index: int) -> Optional[BaseObject]:
        """
        Get a BaseObject instance from the scene by index.

        Args:
            index(int): The index of the BaseObject instance to retrieve.

        Returns:
            The BaseObject instance with the specified index, or None if not found.
        """
        if index < len(self.objects):
            return self.objects[index]
        return None

    def get_index_by_name(self, name: str) -> Optional[int]:
        """
        Get the index of a BaseObject instance from the scene by name.

        Args:
            name(str): The name of the BaseObject instance to retrieve.

        Returns:
            The index of the BaseObject instance with the specified name, or None if not found.
        """
        for i, obj in enumerate(self.objects):
            if obj.name == name and type(obj) == BaseObject:
                return i

        return None

    def get_num_objects(self) -> int:
        """
        Get the number of BaseObject instances in the scene.

        Returns:
            The number of BaseObject instances in the scene.
        """
        return len(self.objects)
