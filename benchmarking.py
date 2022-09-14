import gpflow
import tensorflow_probability as tfp
from gpflow import set_trainable

from gpflow_vgpmp.models.vgpmp import VGPMP
from gpflow_vgpmp.utils.miscellaneous import *
from gpflow_vgpmp.utils.simulator import RobotSimulator

gpflow.config.set_default_float(np.float64)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpflow.config.Config(jitter=1e-6)

if __name__ == '__main__':
    env = RobotSimulator()
    params = env.get_simulation_params()
    planner_params = params.planner
    robot_params = params.robot
    scene_params = params.scene
    num_steps = planner_params["num_steps"]
    robot = env.robot
    scene = env.scene
    sdf = env.sdf

    queries, initial_config_names, initial_config_pose, initial_config_joints, active_joints, sphere_links = create_problems(
        scene_params["problemset"])

    first_query = queries[4]
    start_joints = np.array(first_query[0], dtype=np.float64)
    end_joints = np.array(first_query[1], dtype=np.float64)

    robot.initialise(start_joints, active_joints, sphere_links, initial_config_names, initial_config_pose,
                     initial_config_joints)

    dof = robot.dof

    X = tf.convert_to_tensor(np.array(
        [np.full(7, i) for i in np.linspace(0, 1, 30)], dtype=np.float64))
    y = tf.concat([start_joints.reshape((1, dof)), end_joints.reshape((1, dof))], axis=0)
    Xnew = tf.convert_to_tensor(np.array(
        [np.full(7, i) for i in np.linspace(0, 1, 20)], dtype=np.float64))

    # < ----------------- parameters --------------->

    num_data, num_output_dims = y.shape
    planner = VGPMP.initialize(num_data=num_data,
                               num_output_dims=num_output_dims,
                               num_spheres=planner_params["num_spheres"],
                               num_inducing=planner_params["num_inducing"],
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
                               parameters=params.robot
                               )

    planner.likelihood.variance.prior = tfp.distributions.Normal(gpflow.utilities.to_default_float(0.1),
                                                                 gpflow.utilities.to_default_float(0.005))

    set_trainable(planner.likelihood.alpha, False)
    set_trainable(planner.kernel.kernels, False)
    set_trainable(planner.inducing_variable, False)

    # some_config = tf.convert_to_tensor(start_joints.reshape((1, dof)), dtype=tf.float64)

    # sample_dim = 1
    # K = tf.map_fn(lambda i: planner.likelihood.sampler._fk_cost(tf.reshape(
    #     some_config[i], (-1, 1))), tf.range(sample_dim), fn_output_signature=tf.float64, parallel_iterations=None)

    # _joint_pos = robot.compute_joint_positions(start_joints.reshape((dof, 1)))
    # draw_active_config(robot, _joint_pos, 1, env.sim.physicsClient)

    # _link_pos = robot.get_link_world_pos(robot.active_link_idx)
    # draw_active_config(robot, _link_pos, 2, env.sim.physicsClient)

    # link_joint_pos = _joint_pos + robot.joint_link_offsets
    # draw_active_config(robot, link_joint_pos, 0, env.sim.physicsClient)

    # K = tf.squeeze(K)
    # prev = K[0]
    # for i in range(1, K.shape[0]):
    #     cur = K[i]
    #     p.addUserDebugLine(prev, cur, lineColorRGB=[1, 0, 1],
    #                        lineWidth=5.0, lifeTime=0, physicsClientId=env.sim.physicsClient)
    #     prev = cur

    # p.resetBasePositionAndOrientation(robot.robot_model, posObj=[-0.5, 0, 0], ornObj=[0, 0, 0, 1])
    # env.loop()

    training_loop(model=planner, num_steps=num_steps, data=X, optimizer=planner.optimizer)

    print(planner.likelihood.variance)
    joint_configs = planner.sample_from_posterior(Xnew)

    # write_parameter_to_file(planner.q_sqrt, 'q_sqrt')
    # write_parameter_to_file(planner.q_mu, 'q_mu')

    print(joint_configs)
    robot.enable_collision_active_links(-1)
    robot.set_joint_position(start_joints)
    link_pos = robot.compute_joint_positions(np.reshape(start_joints, (7, 1)))
    link_pos += robot.joint_link_offsets

    EE = [link_pos[-1]]
    prev = link_pos[-1]
    for joint_config in joint_configs:
        link_pos = robot.compute_joint_positions(np.reshape(joint_config, (7, 1)))
        link_pos += robot.joint_link_offsets
        link_pos = np.array(link_pos[-1])
        p.addUserDebugLine(prev, link_pos, lineColorRGB=[0, 0, 1],
                           lineWidth=5.0, lifeTime=0, physicsClientId=env.sim.physicsClient)
        time.sleep(2)
        prev = link_pos
        EE.append(link_pos)

    EE = np.array(EE)  # .reshape((12, 3))
    print(EE, EE.shape)
    plot(EE, "plot_1")

    robot.move_to_ee_config(joint_configs)
