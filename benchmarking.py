import gpflow
import tensorflow_probability as tfp
import numpy as np
from gpflow import set_trainable
from gpflow_vgpmp.models.vgpmp import VGPMP
from gpflow_vgpmp.utils.miscellaneous import *
from gpflow_vgpmp.utils.simulator import RobotSimulator

gpflow.config.set_default_float(np.float64)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpflow.config.Config(jitter=1e-6)

if __name__ == '__main__':
    env = RobotSimulator()
    p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=-75, cameraPitch=-45, cameraTargetPosition=[0, 0, 0])

    # env.loop()
    params = env.get_simulation_params()
    planner_params = params.planner
    robot_params = params.robot
    scene_params = params.scene
    num_steps = planner_params["num_steps"]
    robot = env.robot
    scene = env.scene
    sdf = env.sdf
    queries, initial_config_names, initial_config_pose, initial_config_joints, active_joints = create_problems(
        scene_params["problemset"], robot_name=robot_params["robot_name"])
    query_index = 5

    # Problemsets
    # start_joints = np.array(queries[6], dtype=np.float64)
    # end_joints = np.array(queries[4], dtype=np.float64)

    # Bookshelves + table
    # start_joints = np.array([-0.08271578, -0.05581101, -0.06679014, -0.51343344, -0.05400771,  2.15320285, -1.60573894], dtype=np.float64)
    # start_joints = np.array([ 2.89, 0.50844082,  2.73965633, -1.98486023,  1.79541297,  1.66740883, 1.46448257], dtype=np.float64)
    # end_joints = np.array([ 0.01263066, -0.43801201,  1.67350792, -1.3127274,  -0.11203733,  2.51337125, -1.744899  ], dtype=np.float64)
    # start_joints = np.array([-1.48461325,  1.75057515,  2.55503336, -2.0396316,  -0.80634439,  2.58855503, 2.76517176], dtype=np.float64)
    # start_joints = np.array([0] * 7, dtype=np.float64)

    # Tight env
    # start_joints = np.array([ 0.98692418,  0.33437095, -0.65302636, -1.5013953,  -1.36486178,  2.687824, -0.62724724], dtype=np.float64)
    # end_joints = np.array([-0.25374755,  0.64612583, -0.76724315, -0.90762323, -0.68691582,  1.96506413, -0.61479132], dtype=np.float64)

    # Industrial
    # start_joints = np.array([-1.38294485, 0.61212638, -1.31441932, -0.22121811, 1.1110808, 1.38602424,
    #                          0.81529816], dtype=np.float64)
    start_joints = np.array([-1.16439096, 0.65851989, -1.24320806, -0.14266402, 1.18478857, 1.56439271,
                             1.50726637], dtype=np.float64)
    end_joints = np.array([-0.51877685, 0.38124115, 0.7164529, -1.1444525, -0.15082004, 1.8269117,
                            2.8963512], dtype=np.float64)
    # end_joints = np.array([[0.10218915, 0.67604317, -0.39735951, -0.3600791, -1.42869601, 2.84581188,
    #                         -1.26557614]], dtype=np.float64)
    sphere_links = robot_params["spheres_over_links"]

    robot.initialise(start_joints, active_joints, sphere_links, initial_config_names, initial_config_pose,
                     initial_config_joints, 0)
    print(robot.compute_joint_positions(end_joints.reshape(7, -1))[0][-1])
    dof = robot.dof

    X = tf.convert_to_tensor(np.array(
        [np.full(7, i) for i in np.linspace(0, 1, 70)], dtype=np.float64))
    y = tf.concat([start_joints.reshape((1, dof)), end_joints.reshape((1, dof))], axis=0)
    print(y)

    Xnew = tf.convert_to_tensor(np.array(
        [np.full(7, i) for i in np.linspace(0, 1, 50)], dtype=np.float64))
    #
    # # < ----------------- parameters --------------->

    start_pos, start_mat = robot.compute_joint_positions(start_joints.reshape(7, -1))
    end_pos, end_mat = robot.compute_joint_positions(end_joints.reshape(7, -1))

    print("Start pos: ", start_pos)
    print("End pos: ", end_pos)

    num_data, num_output_dims = y.shape
    planner = VGPMP.initialize(num_data=num_data,
                               num_output_dims=num_output_dims,
                               num_spheres=planner_params["num_spheres"],
                               num_inducing=planner_params["num_inducing"],
                               num_samples=planner_params["num_samples"],
                               sigma_obs=planner_params["sigma_obs"],
                               alpha=planner_params["alpha"],
                               learning_rate=planner_params["learning_rate"],
                               lengthscale=planner_params["lengthscale"],
                               offset=scene_params["object_position"],
                               joint_constraints=robot_params["joint_constraints"],
                               velocity_constraints=robot_params["velocity_constraints"],
                               rs=robot.rs,
                               query_states=y,
                               sdf=sdf,
                               robot=robot,
                               num_latent_gps=dof,
                               parameters=params.robot,
                               q_mu=None,
                               # whiten=False,
                               )

    # planner.likelihood.variance.prior = tfp.distributions.Normal(gpflow.utilities.to_default_float(0.0005),
    #                                                              gpflow.utilities.to_default_float(0.005))

    set_trainable(planner.alpha, False)
    # set_trainable(planner.kernel.kernels, True)
    set_trainable(planner.kernel.kernel.variance, False)
    set_trainable(planner.kernel.kernel.lengthscales, True)
    set_trainable(planner.inducing_variable, False)
    # for kern in planner.kernel.kernels:
    #     set_trainable(kern.variance, False)

    training_loop(model=planner, num_steps=num_steps, data=X, optimizer=planner.optimizer)
    joint_configs, samples = planner.sample_from_posterior(Xnew)
    tf.print(planner.likelihood.variance, summarize=-1)
    robot.enable_collision_active_links(-1)
    robot.set_joint_position(start_joints)
    link_pos, _ = robot.compute_joint_positions(np.reshape(start_joints, (7, 1)))
    EE = [link_pos[-1]]
    prev = link_pos[-1]

    for joint_config in joint_configs:
        link_pos, _ = robot.compute_joint_positions(np.reshape(joint_config, (7, 1)))
        link_pos = np.array(link_pos[-1])
        p.addUserDebugLine(prev, link_pos, lineColorRGB=[0, 0, 1],
                           lineWidth=5.0, lifeTime=0, physicsClientId=env.sim.physicsClient)

        p.addUserDebugLine(prev, link_pos, lineColorRGB=[1, 0, 0],
                           lineWidth=5.0, lifeTime=0, physicsClientId=env.sim.physicsClient)

        time.sleep(0.1)
        prev = link_pos
        EE.append(link_pos)

    for sample in samples:
        link_pos, _ = robot.compute_joint_positions(np.reshape(start_joints, (7, 1)))
        prev = link_pos[-1]

        for joint_config in sample:
            link_pos, _ = robot.compute_joint_positions(np.reshape(joint_config, (7, 1)))
            link_pos = np.array(link_pos[-1])
            p.addUserDebugLine(prev, link_pos, lineColorRGB=[1, 0, 0],
                            lineWidth=5.0, lifeTime=0, physicsClientId=env.sim.physicsClient)

            prev = link_pos
    link_pos, _ = robot.compute_joint_positions(np.reshape(end_joints, (7, 1)))
    link_pos = np.array(link_pos[-1])
    p.addUserDebugLine(prev, link_pos, lineColorRGB=[0, 0, 1],
                       lineWidth=5.0, lifeTime=0, physicsClientId=env.sim.physicsClient)
    EE.append(link_pos)
    EE = np.array(EE)  # .reshape((12, 3))
    print(EE, EE.shape)
    print(f" alpha {planner.alpha}")
    print(f" lengthscale {planner.kernel.kernel.lengthscales}")
    print(f" variance {planner.kernel.kernel.variance}")
    time.sleep(5)
    print("joint configs", joint_configs)
    print("y", y)
    robot.move_to_ee_config(joint_configs)
