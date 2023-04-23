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

    if graphics_params["debug_joint_positions"]:
        config = np.array(queries[0][0])
        start_pos, start_mat = robot.compute_joint_positions(config.reshape(robot.dof, -1),
                                                             robot_params["craig_dh_convention"])

        for pos in start_pos:
            aux_pos = np.array(pos).copy()
            aux_pos[2] += 0.05
            p.addUserDebugLine(pos, aux_pos, lineColorRGB=[0, 0, 1],
                               lineWidth=5.0, lifeTime=0, physicsClientId=env.sim.physicsClient)

    if graphics_params["debug_joint_positions"] and not graphics_params["debug_sphere_positions"]:
        base_pos, base_rot = p.getBasePositionAndOrientation(robot.robot_model)

        p.resetBasePositionAndOrientation(robot.robot_model, (base_pos[0] - 0.5, base_pos[1], base_pos[2]), base_rot)
        env.loop()  # The .loop() function is needed to visualize the joint positions.
        # It is an infinite while loop that is broken when you press the "q" key.
        # It can also give you the current joint configuration of the robot when you
        # press the "a" key.
        # If you are also debugging the sphere positions, you can skip this.

    # ENDING DEBUGING CODE FOR VISUALIZING JOINTS

    # print(planner_params)
    with open("{}_{}.txt".format(robot_params["robot_name"], scene_params["problemset"]), "w") as f:
        for querry in queries:
            f.write(f"{list(querry)}\n")

    # sys.exit()
    
    states = [list() for _ in range(15)]
    states[0] = [ 0.04295548, 0.95584516, -0.96807816, 0.97116162, 0.9778903, 0.65763463, -0.68464669] # top
    states[1] = [ 0.16082985, 1.11182696, -0.92183762, 0.3794195,   1.23 ,       0.47523424, -0.27413472] # top 
    states[2] = [ 0.09952304, 1.09863569, -0.88496722, 0.38292964, 1.23, 0.41536308, -0.38031438] # top
    states[3] = [ 0.10052545, 1.06389854, -1.09858978, 0.48121717, 0.76275836, 1.38780074, 0.79727844] # top
    states[4] = [-0.45014853, 1.59318377, 0.4554682, 0.6065858, -0.38585459, 0.53452102, 0.00784768] # bottom
    states[5] = [-0.34010213,  1.6881081,   0.98402557, 0.51367941, -2.39890266, -0.58455747, 1.01213727] # bottom
    states[6] = [-0.22101804, 1.66367157, 1.09508804, 0.56299024, -2.89040372, -0.59143963, 1.31477334] # bottom
    states[7] = [-0.67729868, 1.64146044, 1.12373694, 0.91912803, -3.17152523, -0.89928808, 1.388017  ] # bottom
    states[8] = [-1.36399638, 1.91753362, 1.32779556, 2.07333031, 0.8333524, 0.08067977, -2.31735325] # bottom
    states[9] = [-0.87877812, 1.64645585, 1.34329545, 1.62880413, 0.84055928, -0.0062247, -2.29039162] # bottom
    states[10] = [ 1.38153424, 1.78324208, 0.18278696, 0.43210283, -1.62168076, 1.01491547, 2.18338891] # table
    states[11] = [ 1.60174351, 1.74358664, 0.12658995, 0.20548551, -1.48280243, 0.92108951, 2.38725579] # table
    states[12] = [ 1.9937845, 1.52197993, 0.44538624, 1.10392873, -1.28498349, 1.32703383, 2.49745328] # table
    states[13] = [-1.29228216, -1.90587936, 1.65480383, 0.20854488, 0.6896924, 0.52053023, -2.4882973 ] # table
    states[14] = [0] * 7
    query_indices = [(3, 10), (10, 4), (4, 11), (11, 13), (13, 12), (12, 5), (5, 8), (8, 16),
                     (16, 9), (9, 5), (5, 15), (15, 12), (12, 6), (6, 7), (7, 14), (14, 3),
                     (3, 9), (9, 4), (4, 13), (13, 8)]
    # (13, 12)   fails # index 4
    # (8, 16)    fails # index 7
    # (9, 5)     fails # index 9
    # (12, 6)    fails # index 12
    # (6, 7)     fails # index 13
    # (7, 14)    fails # index 14
    # (9, 4)     fails # index 18

    if robot_params["robot_name"] == "wam":
        base_pos, base_rot = p.getBasePositionAndOrientation(robot.robot_model)
        p.resetBasePositionAndOrientation(robot.robot_model, (base_pos[0], base_pos[1], -0.346 + base_pos[2]), base_rot)

    elif robot_params["robot_name"] == "ur10":
        base_pos, base_rot = p.getBasePositionAndOrientation(robot.robot_model)
        p.resetBasePositionAndOrientation(robot.robot_model, base_pos, (0, 0, 0, 1))
    # robot.set_curr_config(np.squeeze(states[14]))
    # env.loop()
    # WAM
    not_worked = {11, 12, 13, 19, 21, 44, 33, 42, 43} # just start and end q mu init
    not_worked = {12, 13, 21, 43, 44} # increase the number of iterations to 200 from 130, to make them work, use the mean with the ciung state
    
    # UR10
    # not_worked = {11, 19, 20, 21, 22, 27, 28, 31} # with 7 samples and with middle q_mu init until index 32, excluding it
    # not_worked = {34, 37, 41, 43, 44} # with 20 samples and with middle q_mu init from index 32, including it
    # not_worked = {21, 22, 28, 31} # what is left to not work with 20 samples
    not_worked = {21, 22, 28, 31, 34, 37, 41, 43, 44} # these are the ones that DO NOT work with 20 samples, the rest should be fine
                                                      # index 37 benefits from star and end q mu init
    # to get things to work for ur10 use the first picture params until index 32, then use the second picture params from index 32
    # these are the params that worked for the first two not_worked indices
    # then for the first not_worked index, increase the number of samples to 20
    nono = []
    bef_32_20_samples = {11, 19, 20, 27}

    # KUKA
    not_worked = {2, 3, 4, 7, 9, 11, 12, 13, 16, 18, 19, 20, 21, 24, 26, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 48, 50, 51, 52, 53}
    not_worked = {4, 13, 19, 21, 33, 40, 41, 42, 43, 44, 52}
    total_solved = 0
    # p.addUserDebugLine([0, 0, 1], [0, 0, 0], rgbaColor=[1, 0, 0], 
    #                                lineWidth=5.0, lifeTime=0, physicsClientId=env.sim.physicsClient)
    # env.loop()
    # WAM
    new_not_worked = [15, 54, 15, 45, 49, 51, 54, 30, 36, 54, 6, 15, 54, 14, 23, 45, 51, 54]
    start_joints = np.array([0] * 7, dtype=np.float64).reshape(1, robot.dof)
    end_joints = np.array([0] * 7, dtype=np.float64).reshape(1, robot.dof)
    planner = create_planner(env=env,
                            robot=robot,
                            sdf=sdf,
                            start_joints=start_joints,
                            end_joints=end_joints,
                            robot_params=robot_params,
                            planner_params=planner_params,
                            scene_params=scene_params,
                            trainable_params=trainable_params,
                            graphics_params=graphics_params)
    # print(gpflow.utilities.parameter_dict(planner))
    weights = gpflow.utilities.parameter_dict(planner)
    weights = gpflow.utilities.deepcopy(weights)#weights.copy()
    # not_worked = {1, 2, 7, 14, 17}
    not_worked = {9, 13, 19}
    total_runs = 5

    for run in range(1):
        for k, (i, j) in enumerate(query_indices):
        # for i, (start_joints, end_joints) in enumerate(queries[2:]):
            # i += 35
            # if i != 54:
            #     continue
            # if run == 0:
            #     if i in {0, 1, 2, 3, 4, 5}:
            #         continue
            # UR10 MADNESS
            # if i == 31:
            #     continue
            # # elif i == 31:
            # #     planner_params["num_samples"] = 10
            # elif i == 37:
            #     continue
            #     # planner_params["num_samples"] = 20
            # elif i < 32:
            #     if i in bef_32_20_samples:
            #         planner_params["num_samples"] = 20
            #     else:
            #         planner_params["num_samples"] = 7
            # else:
            #     planner_params["num_samples"] = 20
            
            # LAB STUFF
            # if k == 3:
            #     planner_params["sigma_obs"] = 0.05
            # else:
            #     planner_params["sigma_obs"] = 0.005
            # if k not in not_worked:
            #     continue
            start_joints = np.array(states[i-3], dtype=np.float64).reshape(1, robot.dof)
            end_joints = np.array(states[j-3], dtype=np.float64).reshape(1, robot.dof)
            robot.set_curr_config(np.squeeze(start_joints))
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
                                            planner=planner,
                                            weights=weights,
                                            run=run,
                                            k=k)
            total_solved += solved
            if not solved:
                print(f"Failed to solve problem {k}")
                nono.append(k)
            p.removeAllUserDebugItems()
    print(nono)
    print(f"Average total solved: {total_solved / total_runs} out of {len(query_indices)}")
    time.sleep(10)