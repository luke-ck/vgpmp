import asyncio
import queue
import threading
from collections import deque
from pathlib import Path
from typing import List, Optional

import pybullet as p
from .bullet_object import BaseObject

__all__ = 'simulation'


def get_bullet_key_from_value():
    """ Get the key from the value in the bullet dictionary """
    return {value: key for key, value in p.__dict__.items() if key.startswith("B3G")}


class SimulationThread(threading.Thread):
    def __init__(self, graphic_params: dict):
        super().__init__()
        self.graphic_params = graphic_params
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.thread_ready_event = threading.Event()
        self.client = None

    async def check_connection(self):
        assert self.client is not None, "Client is not initialized"
        while not self.stop_event.is_set():
            if p.getConnectionInfo(self.client)['isConnected'] == 0:
                self.stop_event.set()
                return
            if self.stop_event.is_set():
                return
            await asyncio.sleep(0.01)
        self.client = None

    async def wait_key_press(self):
        async for key, is_return_pressed in self.await_key_press():
            self.result_queue.put(key)
            if is_return_pressed:
                self.stop_event.set()
                return

    async def await_key_press(self):
        while True:
            keys = p.getKeyboardEvents()
            if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
                yield keys.keys(), False
                return

            if p.B3G_RETURN in keys and keys[p.B3G_RETURN] & p.KEY_WAS_TRIGGERED:
                yield keys.keys(), True
                return

            if self.stop_event.is_set():
                yield "STOP", True
                return

            await asyncio.sleep(0.01)

    async def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        tasks = [self.check_connection(), self.wait_key_press()]
        try:
            await asyncio.gather(*[asyncio.create_task(task) for task in tasks])
        except asyncio.CancelledError:
            pass
        finally:
            loop.close()

    def initialize(self):

        gravity_constant = -9.81

        if self.graphic_params["visuals"]:
            self.client = p.connect(p.GUI, options="--width=960 --height=540")
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.setRealTimeSimulation(enableRealTimeSimulation=1)
            p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)
        else:
            self.client = p.connect(p.DIRECT)
        p.resetSimulation()
        p.setGravity(0, 0, gravity_constant)
        self.thread_ready_event.set()  # Notify the main thread that the simulation thread is ready

    def stop(self):
        self.stop_event.set()
        if self.client is not None:
            p.disconnect(self.client)
            self.client = None


class Simulation:
    def __init__(self, graphic_params):
        self.is_initialized = None
        self.result_queue = None
        self.scene = None
        self.result_queue = queue.Queue()
        self.graphic_params = graphic_params
        self.simulation_thread = SimulationThread(graphic_params)
        # self.start_simulation_thread()

    def initialize(self):
        self.simulation_thread.initialize()
        self.simulation_thread.thread_ready_event.wait()
        print("Connecton to pybullet backend started")
        self.is_initialized = True

    def stop_simulation_thread(self):
        self.simulation_thread.stop()
        assert self.simulation_thread.client is None, "Client is not None"
        assert self.simulation_thread.is_alive() is False, "Simulation thread is still alive"
        print("Simulation thread stopped")

    def check_events(self):
        """ TODO: do stuff with the simulation thread"""
        # queue_result = self.result_queue.get().result()
        # queue_size = self.result_queue.qsize()
        # print("Queue size: ", queue_size)
        # print("Queue result: ", queue_result)
        # for key in queue_result:
        #     if key == p.B3G_RETURN:
        #         self.simulation_thread.stop()
        #         self.simulation_thread.join()
        #         print("Simulation thread stopped")
        #     else:
        #         print("Something went wrong with the simulation thread")
        pass

    def check_simulation_thread_health(self):
        """ check if the simulation thread is still alive """
        return self.simulation_thread.is_alive()


class Scene:
    """
    Class for managing a scene of BaseObjects.
    """

    def __init__(self, config: dict):
        """
        Create a new Scene instance.
        """
        self.is_initialized = None
        self.config = config
        self.client = None
        # FIFO queue for storing the objects
        self.objects = deque()

    def initialize(self, client):
        self.client = client
        self.add_object(name="plane", path=None, position=None, orientation=None)
        self.add_object(name="environment",
                        path=self.config["environment_path"],
                        position=self.config["position"],
                        orientation=self.config["orientation"])
        if self.config["objects"] is not None and self.config["objects"] is not []:
            # TODO: add support for multiple objects
            for obj_path in self.config["object_path"]:
                self.add_object(name="object",
                                path=obj_path,
                                position=self.config["position"],
                                orientation=self.config["orientation"])
        self.is_initialized = True

    def add_object(self, name: str, path: Optional[str], position: Optional[List], orientation: Optional[List]):
        """
        Add a new BaseObject instance to the scene.
        """
        path = Path(path) if type(path) == str else path  # Convert path to Path object if it is a string
        assert self.client is not None, "No client is connected to the backend"
        obj = BaseObject(name=name, path=path, position=position, orientation=orientation, client=self.client)
        self.objects.appendleft(obj)

    def remove_object(self, obj: BaseObject, index: Optional[int] = None):
        """
        Remove a BaseObject instance from the scene.
        """
        obj.remove()
        if index is not None:
            self.objects.remove(self.objects[index])
        # remove last occurence of obj from list
        for item in self.objects:
            if item == obj:
                self.objects.remove(item)
                break
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
