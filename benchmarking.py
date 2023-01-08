import gpflow
import numpy as np
from gpflow_vgpmp.utils.miscellaneous import *
from gpflow_vgpmp.utils.simulator import RobotSimulator

gpflow.config.set_default_float(np.float64)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpflow.config.Config(jitter=1e-6)

if __name__ == '__main__':
    env = RobotSimulator()
    p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=-75, cameraPitch=-45, cameraTargetPosition=[0, 0, 0])

    params = env.get_simulation_params()
    planner_params = params.planner
    robot_params = params.robot
    scene_params = params.scene
    trainable_params = params.trainable_params
    graphics_params = params.graphics
    num_steps = planner_params["num_steps"]
    sphere_links = robot_params["spheres_over_links"]
    active_joints = robot_params["active_joints"]
    robot = env.robot
    scene = env.scene
    sdf = env.sdf
    queries, initial_config_names, initial_config_joints = create_problems(
        scene_params["problemset"], robot_name=robot_params["robot_name"])

    # Problemsets
    # start_joints = np.array(queries[11], dtype=np.float64) # 11, 18
    # end_joints = np.array(queries[14], dtype=np.float64)

    # Bookshelves + table
    # start_joints = np.array([-0.08271578, -0.05581101, -0.06679014, -0.51343344, -0.05400771,  2.15320285, -1.60573894], dtype=np.float64)
    # start_joints = np.array([ 2.89, 0.50844082,  2.73965633, -1.98486023,  1.79541297,  1.66740883, 1.46448257], dtype=np.float64)
    # end_joints = np.array([ 0.01263066, -0.43801201,  1.67350792, -1.3127274,  -0.11203733,  2.51337125, -1.744899  ], dtype=np.float64)
    # start_joints = np.array([-1.48461325,  1.75057515,  2.55503336, -2.0396316,  -0.80634439,  2.58855503, 2.76517176], dtype=np.float64)

    # Tight env
    # start_joints = np.array([ 0.98692418,  0.33437095, -0.65302636, -1.5013953,  -1.36486178,  2.687824, -0.62724724], dtype=np.float64)
    # end_joints = np.array([-0.25374755,  0.64612583, -0.76724315, -0.90762323, -0.68691582,  1.96506413, -0.61479132], dtype=np.float64)

    # Industrial
    # start_joints = np.array([-1.38294485, 0.61212638, -1.31441932, -0.22121811, 1.1110808, 1.38602424,
    #                          0.81529816], dtype=np.float64)
    # start_joints = np.array([-1.16439096, 0.65851989, -1.24320806, -0.14266402, 1.18478857, 1.56439271,
    #                          1.50726637], dtype=np.float64)
    # end_joints = np.array([-0.51877685, 0.38124115, 0.7164529, -1.1444525, -0.15082004, 1.8269117,
    #                         2.8963512], dtype=np.float64)
    # end_joints = np.array([[0.10218915, 0.67604317, -0.39735951, -0.3600791, -1.42869601, 2.84581188,
    #                         -1.26557614]], dtype=np.float64)
    
    sphere_links = robot_params["spheres_over_links"]
    start_joints = np.array(queries[0], dtype=np.float64)
    robot.initialise(start_joints, active_joints, sphere_links, initial_config_names, initial_config_joints, 0)

    # DEBUGING CODE FOR VISUALIZING JOINTS

    # This part of the code takes the start_joints confuguration that is above
    # and visualizes the joint positions by drawing a blue line from the joint position to 
    # +0.15 in the z direction. Change this to a lower value if you want to see the joint positions better.
    # If you are debugging the sphere positions also, make sure to match this with the
    # same joint configuration that is in the first tuple of the query_indices list.
    # The first element of the first tuple is the joint configuration that you visualize.

    if graphics_params["debug_joint_positions"]:
        start_pos, start_mat = robot.compute_joint_positions(start_joints.reshape(robot.dof, -1), robot_params["craig_dh_convention"])

        for pos in start_pos:
            aux_pos = np.array(pos).copy()
            aux_pos[2] += 0.05
            p.addUserDebugLine(pos, aux_pos, lineColorRGB=[0, 0, 1],
                            lineWidth=5.0, lifeTime=0, physicsClientId=env.sim.physicsClient)
    
    if graphics_params["debug_joint_positions"] and not graphics_params["debug_sphere_positions"]:
        base_pos, base_rot = p.getBasePositionAndOrientation(robot.robot_model)

        p.resetBasePositionAndOrientation(robot.robot_model, (base_pos[0]-0.5, base_pos[1], base_pos[2]), base_rot)
        env.loop() # The .loop() function is needed to visualize the joint positions.
                   # It is an infinite while loop that is broken when you press the "q" key.
                   # It can also give you the current joint configuration of the robot when you
                   # press the "a" key.
                   # If you are also debugging the sphere positions, you can skip this.

    # ENDING DEBUGING CODE FOR VISUALIZING JOINTS
    
    total_solved = 0
    query_indices = [(7, 5)] # [(3, 10), (10, 4), (4, 11), (11, 13), (13, 12), (12, 5), (5, 8), (8, 16), 
                     # (16, 9), (9, 5), (5, 15), (15, 12), (12, 6), (6, 7), (7, 14), (14, 3), 
                     # (3, 9), (9, 4), (4, 13), (13, 8)]
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
    # env.loop()
    for i, j in query_indices:
        start_joints = np.array(queries[i], dtype=np.float64)
        end_joints = np.array(queries[j], dtype=np.float64)
        robot.set_curr_config(start_joints)
        solved = solve_planning_problem(env=env, robot=robot, sdf=sdf, start_joints=start_joints, end_joints=end_joints,
                            robot_params=robot_params, planner_params=planner_params, scene_params=scene_params, 
                            trainable_params=trainable_params, graphics_params=graphics_params)
        total_solved += solved

    print(f"Planner solved {total_solved} / {len(query_indices)} problems")
    time.sleep(10)