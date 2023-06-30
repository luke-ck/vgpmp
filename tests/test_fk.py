# this test checks the equivalence of the numpy and tensorflow implementations
import numpy as np
import tensorflow as tf

from data.problemsets.problemset import create_problems
from gpflow_vgpmp.utils.robot import Robot
from gpflow_vgpmp.utils.sampler import Sampler
from gpflow_vgpmp.utils.simulation import ParameterLoader
from gpflow_vgpmp.utils.simulator import RobotSimulator


def test_tf_fk_cost():
    env = RobotSimulator(parameter_file_path='./test_params.yaml')
    robot = env.robot
    scene = env.scene
    sdf = env.sdf

    params = env.get_simulation_params()
    robot_params = params.robot_params

    queries, planner_params, joint_names, default_pose, default_robot_pos_and_orn = create_problems(
        problemset_name=robot_params['problemset'], robot_name=robot_params['robot_name'])

    default_pose = np.array([0] * robot_params['dof']).reshape(robot_params['dof'], 1)
    start_config = np.array([0] * robot_params['dof']).reshape(robot_params['dof'], 1)
    robot.initialise(default_robot_pos_and_orn=default_robot_pos_and_orn,
                     robot_params=robot_params,
                     start_config=start_config,
                     joint_names=joint_names,
                     default_pose=default_pose,
                     benchmark=True)

    robot_config = {"sphere_offsets": robot.sphere_offsets,
                    "num_spheres_list": robot.num_spheres,
                    "dof": robot.dof,
                    "sphere_link_interval": robot.sphere_link_interval,
                    "base_pose": robot.base_pose,
                    'robot_name': params.robot_params['robot_name']
                    }

    sampler = Sampler(params.robot_params, robot_config)
    joints = robot.get_curr_config().reshape(7, 1)
    position = sampler._fk_cost(joints)
    # joints = robot.get_curr_config().reshape(7, 1)
    # position = sampler._fk_cost(joints)
    # print(position)


if __name__ == "__main__":
    test_tf_fk_cost()
