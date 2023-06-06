import sys
import os
from bunch import Bunch
import tensorflow as tf
from gpflow_vgpmp.utils.bullet_object import Object
from gpflow_vgpmp.utils.robot import Robot
from gpflow_vgpmp.utils.sdf_utils import SignedDensityField
from gpflow_vgpmp.utils.simulation import Simulation
import pybullet as p
# ---------------Exports
__all__ = 'simulator'

class Mesh:
    """Load a mesh into the simulation. Mass of the mesh is set to 0 so that it does not fall under gravity."""
    def __init__(
        
        self, path = "/home/freshpate/vgpmp/data/scenes/lab/lab.obj", 
        scale = [1, 1, 1], 
        shift = [0, 0, 0],
        positionXYZ = [0, 0, 1], 
        rgbColor = [0.58, 0.29, 0.0],
        orientationXYZ = [0, 0, 0],
        alpha = 1
        ):
        self.path = path
        self.scale = scale
        self.position = positionXYZ
        self.orientation = orientationXYZ
        self.shift = shift
        self.rgbColor = list(rgbColor)

        self.visualShapeId = p.createVisualShape(
            shapeType=p.GEOM_MESH, 
            fileName=self.path,
            rgbaColor=self.rgbColor+[alpha], 
            specularColor=[0.0, 0.0, 0.0],
            meshScale=self.scale, 
            visualFramePosition=self.shift
            )
        
        self.collisionShapeId = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=self.path, 
            meshScale=self.scale
            )
        
        self.bodyId = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=self.collisionShapeId,
            baseVisualShapeIndex=self.visualShapeId,
            basePosition=self.position,
            baseOrientation=p.getQuaternionFromEuler(self.orientation)
            )
    
    def getVisualShapeId(self):
        return self.visualShapeId

    def getCollisionShapeId(self):
        return self.collisionShapeId
    
    def getBodyId(self):
        return self.bodyId

class RobotSimulator:
    def __init__(self):
        self.sim = Simulation()
        self.plane = Object(name="plane")
        self.robot = Robot(self.sim.robot_params)
        self.sdf = SignedDensityField.from_sdf(self.sim.scene_params["sdf_path"])
        self.scene = Mesh(positionXYZ=self.sim.scene_params["object_position"])
        # self.scene = Object(name="scene",
        #                     path=self.sim.scene_params["object_path"],
        #                     position=self.sim.scene_params["object_position"])
        texture_id = p.loadTexture(os.path.expanduser('~') + "/vgpmp/data/scenes/lab/pringles/textured.png")

        if self.sim.scene_params["problemset"] == "lab":
            self.pringles = Object(name="pringles",
                                   path=os.path.expanduser('~') + "/vgpmp/data/scenes/lab/pringles/pringles.urdf",
                                   position=[-0.005, 0.485, 0.85])
        p.changeVisualShape(self.pringles.ID, -1, textureUniqueId=texture_id)
        # p.connect(self.sim.ph)
        
        self.sett = set([i for i in range(100)])
    def get_simulation_params(self) -> Bunch:
        return self.sim.get_params()

    def loop(self, planner=None):
        exit = False
        while not exit:
            action = input("Enter action: ")
            if action == "q":
                exit = True
            elif action == 'c':
                print(f"Current config is :{self.robot.get_curr_config()}")
            elif action == 'sdf':
                if planner is not None:
                    self.get_rt_sdf_grad(planner)
                else:
                    print("There was no planner given")
            elif action == 'fk':
                if planner is not None:
                    joints = self.robot.get_curr_config()
                    tf.print(planner.debug_likelihood(tf.reshape(joints, (1, 1, 7))))
                else:
                    print("There was no planner given")
            else:
                print(f"Current config is :{self.robot.get_curr_config(int(action))}")

    def get_rt_sdf_grad(self, planner):
        """
        Get the signed distance gradient of the current robot configuration and print it
        """
        joints = self.robot.get_curr_config().reshape(7, 1)
        position = planner.likelihood.sampler._fk_cost(joints)
        print(planner.likelihood._signed_distance_grad(position))
