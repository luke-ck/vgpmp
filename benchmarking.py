from pathlib import Path

from gpflow_vgpmp.utils.miscellaneous import *
from gpflow_vgpmp.utils.simulation_manager import SimulationManager
import os

# set export TF_CPP_MIN_LOG_LEVEL=2 when running for your sanity

gpflow.config.set_default_float(np.float64)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpflow.config.Config(jitter=1e-6)

if __name__ == '__main__':
    parameter_file_path = Path(get_root_package_path()) / "parameters.yaml"
    env = SimulationManager(file_path=parameter_file_path)

    p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=-75, cameraPitch=-45, cameraTargetPosition=[0, 0, 0])

    # DEBUGGING CODE FOR VISUALIZING JOINTS
    # env.loop()
    # This part of the code takes the start_joints configuration that is above
    # and visualizes the joint positions by drawing a blue line from the joint position to 
    # +0.15 in the z direction. Change this to a lower value if you want to see the joint positions better.
    # If you are debugging the sphere positions also, make sure to match this with the
    # same joint configuration that is in the first tuple of the query_indices list.
    # The first element of the first tuple is the joint configuration that you visualize.

    # TODO : Make this a function in the simulation_manager.py file
    # if graphics_params["debug_joint_positions"]:
    #     config = np.array(queries[0][0])
    #     start_pos, start_mat = robot.compute_joint_positions(config.reshape(robot.dof, -1))
    #
    #     for pos in start_pos:
    #         aux_pos = np.array(pos).copy()
    #         aux_pos[2] += 0.05
    #         p.addUserDebugLine(pos, aux_pos, lineColorRGB=[0, 0, 1],
    #                            lineWidth=5.0, lifeTime=0, physicsClientId=env.simulation.client)
    #
    # if graphics_params["debug_joint_positions"] and not graphics_params["debug_sphere_positions"]:
    #     base_pos, base_rot = p.getBasePositionAndOrientation(robot.robot_model)
    #
    #     p.resetBasePositionAndOrientation(robot.robot_model, (base_pos[0] - 0.5, base_pos[1], base_pos[2]), base_rot)
    #     env.loop()  # The .loop() function is needed to visualize the joint positions.
    # It is an infinite while loop that is broken when you press the "q" key.
    # It can also give you the current joint configuration of the robot when you
    # press the "a" key.
    # If you are also debugging the sphere positions, you can skip this.

    # ENDING DEBUGGING CODE FOR VISUALIZING JOINTS


    # PLEASE DO NOT DELETE THESE LINES.
    # They are imporant because there is a slight mismatch between our robot urdf files and the Forward Kinematics.
    # Our convention was to set the "base" of the FK computation as the position of the first joint in the chain.
    # 
    # For WAM, that -0.346 is the height of the support of the robot (the thing the first joint is positioned on).
    # For UR10, the robot needs to be mirrored in the XY plane. Again, due to our urdf.

    if env.config['robot_params']['robot_name'] == "wam":
        base_pos, base_rot = p.getBasePositionAndOrientation(env.robot.robot_model)
        p.resetBasePositionAndOrientation(env.robot.robot_model, (base_pos[0], base_pos[1], -0.346 + base_pos[2]), base_rot)
    
    elif env.config['robot_params']['robot_name'] == "ur10":
        base_pos, base_rot = p.getBasePositionAndOrientation(env.robot.robot_model)
        p.resetBasePositionAndOrientation(env.robot.robot_model, base_pos, (0, 0, 0, 1))


    queries = env.config['scene_params']['queries']
    total_solved = 0
    total_runs = 5
    failed_indices = []

    for _ in range(total_runs):

        for i, (start_joints, end_joints) in enumerate(queries):
            start_joints = np.array(start_joints, dtype=np.float64).reshape(1, env.robot.dof)
            end_joints = np.array(end_joints, dtype=np.float64).reshape(1, env.robot.dof)
            env.robot.set_current_joint_config(np.squeeze(start_joints))
            # env.loop()
            env.robot.set_joint_motor_control(np.squeeze(start_joints), 300, 0.5)
            p.stepSimulation()

            solved, trajectory = solve_planning_problem(env=env,
                                                        start_joints=start_joints,
                                                        end_joints=end_joints)
            print(env.simulation.check_simulation_thread_health())
            total_solved += solved
            if not solved:
                print(f"Failed to solve problem {i}")
                failed_indices.append(i)
            p.removeAllUserDebugItems()
    print(failed_indices)
    print(f"Average total solved: {total_solved / total_runs} out of {len(queries)}")

    time.sleep(10)
    # Disconnect from the simulation
    env.simulation.stop_simulation_thread()
