import gpflow
import numpy as np
from gpflow_vgpmp.utils.miscellaneous import *
from gpflow_vgpmp.utils.simulator import RobotSimulator

gpflow.config.set_default_float(np.float64)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpflow.config.Config(jitter=1e-6)

if __name__ == '__main__':
    env = RobotSimulator()
    robot = env.robot
    scene = env.scene
    sdf = env.sdf

    p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=-75, cameraPitch=-45, cameraTargetPosition=[0, 0, 0])

    params = env.get_simulation_params()
    robot_params = params.robot
    scene_params = params.scene
    trainable_params = params.trainable_params
    graphics_params = params.graphics

    sphere_links = robot_params["spheres_over_links"]
    active_joints = robot_params["active_joints"]
    problemset = scene_params["problemset"]
    robot_name = robot_params["robot_name"]

    queries, planner_params, joint_names, default_pose, default_robot_pos_and_orn = create_problems(
        problemset=problemset, robot_name=robot_name)

    num_steps = planner_params["num_steps"]

    position, orientation = default_robot_pos_and_orn
    robot.initialise(position=position,
                     orientation=orientation,
                     active_joints=active_joints,
                     sphere_links=sphere_links,
                     start_config=default_pose,
                     joint_names=joint_names,
                     default_pose=default_pose,
                     benchmark=True)

    # env.loop()
    # DEBUGING CODE FOR VISUALIZING JOINTS

    # This part of the code takes the start_joints confuguration that is above
    # and visualizes the joint positions by drawing a blue line from the joint position to 
    # +0.15 in the z direction. Change this to a lower value if you want to see the joint positions better.
    # If you are debugging the sphere positions also, make sure to match this with the
    # same joint configuration that is in the first tuple of the query_indices list.
    # The first element of the first tuple is the joint configuration that you visualize.
    end_planner1 = [0.31109301, -0.72880518, 0.75456588, -1.01347162, 0.92935332, 2.24804542, -0.33061236]
    end_planner2 = [0.86212819, -0.8842489, 0.31054139, -1.70539857, 1.08483577, 2.5299051, -0.64731048]
    end = [0.89061377, -0.40183717, 0.48752105, -1.42820565, 0.75600404, 2.60547999, -0.52784457]
    start_planner1 = [-0.19236134, 0.93622663, -0.8607566, -0.22610378, 0.74095531, 1.76073361, -0.95493156]
    start_planner2 = [ 0.15984699, 0.9143245, -0.76890767, -0.38329052, 1.42599777, 1.90409489, -0.84829781]

    queries = [(start_planner1, end_planner1), (start_planner2, end_planner2)]
    # robot.set_curr_config(np.squeeze(end))
    # env.loop()

    if graphics_params["debug_joint_positions"]:
        config = np.array(end_planner2)
        start_pos, start_mat = robot.compute_joint_positions(config.reshape(robot.dof, -1),
                                                             robot_params["craig_dh_convention"])
        print(start_pos)
        for pos in start_pos:
            aux_pos = np.array(pos).copy()
            aux_pos[2] += 0.05
            p.addUserDebugLine(pos, aux_pos, lineColorRGB=[0, 0, 1],
                               lineWidth=5.0, lifeTime=0, physicsClientId=env.sim.physicsClient)
    # env.loop()
    # p.addUserDebugLine([-0.00472223, 0.28439094, 0.98321232], [-0.00472223, 0.28439094, 1.18321232], lineColorRGB=[0, 0, 1],
    #                            lineWidth=5.0, lifeTime=0, physicsClientId=env.sim.physicsClient)

    # p.addUserDebugLine([-0.04855343,  0.09839402,  0.97370369], [-0.04855343,  0.09839402,  1.07370369], lineColorRGB=[0, 0, 1],
    #                            lineWidth=5.0, lifeTime=0, physicsClientId=env.sim.physicsClient)


    if graphics_params["debug_joint_positions"] and not graphics_params["debug_sphere_positions"]:
        base_pos, base_rot = p.getBasePositionAndOrientation(robot.robot_model)

        p.resetBasePositionAndOrientation(robot.robot_model, (base_pos[0] - 0.5, base_pos[1], base_pos[2]), base_rot)
        env.loop()  # The .loop() function is needed to visualize the joint positions.
        # It is an infinite while loop that is broken when you press the "q" key.
        # It can also give you the current joint configuration of the robot when you
        # press the "a" key.
        # If you are also debugging the sphere positions, you can skip this.

    # ENDING DEBUGING CODE FOR VISUALIZING JOINTS

    total_solved = 0
    if robot_params["robot_name"] == "wam":
        base_pos, base_rot = p.getBasePositionAndOrientation(robot.robot_model)
        p.resetBasePositionAndOrientation(robot.robot_model, (base_pos[0], base_pos[1], -0.346 + base_pos[2]), base_rot)

    elif robot_params["robot_name"] == "ur10":
        base_pos, base_rot = p.getBasePositionAndOrientation(robot.robot_model)
        p.resetBasePositionAndOrientation(robot.robot_model, base_pos, (0, 0, 0, 1))


    total_runs = 1
    failed = []
    for run in range(total_runs):

        # for k, (i, j) in enumerate(query_indices):
        for i, (start_joints, end_joints) in enumerate(queries):

            start_joints = np.array(start_joints, dtype=np.float64).reshape(1, robot.dof)
            end_joints = np.array(end_joints, dtype=np.float64).reshape(1, robot.dof)
            robot.set_curr_config(np.squeeze(start_joints))
            # print(end_joints)
            # env.loop()
            robot.set_joint_motor_control(np.squeeze(start_joints), 300, 0.5)
            p.stepSimulation()
            solved = solve_planning_problem(env=env,
                                            robot=robot,
                                            sdf=sdf,
                                            start_joints=start_joints,
                                            end_joints=end_joints,
                                            robot_params=robot_params,
                                            planner_params=planner_params,
                                            scene_params=scene_params,
                                            trainable_params=trainable_params,
                                            graphics_params=graphics_params,
                                            run=run,
                                            k=i)
            total_solved += solved
            if not solved:
                print(f"Failed to solve problem {i}")
                failed.append(i)
            p.removeAllUserDebugItems()
    print(failed)
    print(f"Average total solved: {total_solved / total_runs} out of {len(queries)}")
    time.sleep(10)