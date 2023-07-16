import asyncio
import queue
import threading

import pybullet as p

__all__ = 'simulation'

from .parameter_loader import ParameterLoader


def get_bullet_key_from_value():
    """ Get the key from the value in the bullet dictionary """
    return {value: key for key, value in p.__dict__.items() if key.startswith("B3G")}


class SimulationThread(threading.Thread):
    def __init__(self):
        super().__init__()
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
        # self.client = None

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
        self.thread_ready_event.wait()

        tasks = [self.check_connection(), self.wait_key_press()]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass

    def initialize(self, graphics_params: dict):

        gravity_constant = -9.81

        if graphics_params["visuals"]:
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
            print(f"Disconnected client {self.client}")
            self.client = None

        if self.is_alive():
            self.join()


class Simulation:
    def __init__(self, config: ParameterLoader):
        self.is_initialized = None
        self.result_queue = None
        # TODO: make the scene be managed by the simulation
        self.scene = None
        self.result_queue = queue.Queue()
        assert config.is_initialized, "ParameterLoader is not initialized"
        self.graphics_params = config.graphics_params
        self.simulation_thread = SimulationThread()
        # self.start_simulation_thread()

    def initialize(self):
        self.simulation_thread.initialize(self.graphics_params)
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


