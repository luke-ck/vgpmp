import itertools
import os
import sys
import time
from contextlib import contextmanager
from gpflow.config import default_float
import numpy as np
import pybullet as p
import tensorflow as tf
from gpflow import set_trainable
from tqdm import tqdm
import tensorflow_probability as tfp
import gpflow
from gpflow_vgpmp.models.vgpmp import VGPMP

# from gpflow_vgpmp.utils.simulator import RobotSimulator

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except RuntimeError:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


@contextmanager
def suppress_stdout():
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


def write_parameter_to_file(model_parameter, filepath):
    param = tf.strings.format("{}\n", model_parameter, summarize=-1)
    tf.io.write_file(f"{filepath}.txt", param)


def create_problems(problemset, robot_name):
    r"""
    For the given problemset and robot names, returns the combination of all possible problems,
    the planner parameters for the given environment and robot, and the
    robot joint names, their default pose and the robot position in world coordinates.
    """
    # Start and end joint angles
    Problemset = import_problemsets(robot_name)
    n_states, states = Problemset.states(problemset)
    print('There are %s total robot positions' % n_states)
    # all possible combinations of 2 pairs
    benchmark = list(itertools.combinations(states, 2))
    print('And a total of %d problems in the %s problemset' %
          (len(benchmark), problemset))
    names = Problemset.joint_names(problemset)
    pose = Problemset.default_pose(problemset)
    planner_params = Problemset.planner_params(problemset)
    robot_pos_and_orn = Problemset.pos_and_orn(problemset)

    return benchmark, planner_params, names, pose, robot_pos_and_orn


def set_scene(robot, active_joints, initial_config_joints, initial_config_names, initial_config_pose):
    r"""

    """
    robot.set_active_joints(active_joints)
    for (name, val) in zip(initial_config_names, initial_config_joints):
        idx = robot.get_joint_idx_from_name(name)
        robot.set_joint_config(idx, val)


def optimization_step(model, closure, optimizer):
    r"""

    """
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        loss = closure()

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


def training_loop(model, data, num_steps):
    r"""

    """
    print("Starting training....")
    # data = tf.broadcast_to(tf.sort(tf.random.uniform((10, 1), 0.0001, 0.9999, dtype=tf.float64), axis=0), [10, 7])
    tf_optimization_step = tf.function(optimization_step)
    closure = model.training_loss_closure(data)
    step_iterator = tqdm(range(num_steps))
    for _ in step_iterator:
        loss = tf_optimization_step(model, closure, model.optimizer)
        step_iterator.set_postfix_str(f"ELBO: {-loss:.3e}")


def init_trainset(grid_spacing_X, grid_spacing_Xnew, degree_of_freedom, start_joints, end_joints, scale=100,
                  end_time=1):
    X = tf.convert_to_tensor(np.array(
        [np.full(degree_of_freedom, i) for i in np.linspace(0, end_time * scale, grid_spacing_X)], dtype=np.float64))

    y = tf.concat([start_joints.reshape((1, degree_of_freedom)), end_joints.reshape((1, degree_of_freedom))], axis=0)
    Xnew = tf.convert_to_tensor(np.array(
        [np.full(degree_of_freedom, i) for i in np.linspace(0, end_time * scale, grid_spacing_Xnew)], dtype=np.float64))
    return X, y, Xnew


def solve_planning_problem(env, robot, sdf, start_joints, end_joints, robot_params, planner_params, scene_params,
                           trainable_params, graphics_params):
    grid_spacing_X = planner_params["time_spacing_X"]
    grid_spacing_Xnew = planner_params["time_spacing_Xnew"]
    dof = robot_params["dof"]
    num_steps = planner_params["num_steps"]
    X, y, Xnew = init_trainset(grid_spacing_X, grid_spacing_Xnew, dof, start_joints, end_joints)
    num_data, num_output_dims = y.shape
    q_mu = np.array(robot_params["q_mu"], dtype=np.float64).reshape(1, dof) if robot_params["q_mu"] != "None" else None
    print("robot_params", robot_params)
    planner = VGPMP.initialize(num_data=num_data,
                               num_output_dims=num_output_dims,
                               num_spheres=robot_params["num_spheres"],
                               num_inducing=planner_params["num_inducing"],
                               num_samples=planner_params["num_samples"],
                               sigma_obs=planner_params["sigma_obs"],
                               alpha=planner_params["alpha"],
                               variance=planner_params["variance"],
                               learning_rate=planner_params["learning_rate"],
                               lengthscales=planner_params["lengthscales"],
                               offset=scene_params["object_position"],
                               joint_constraints=robot_params["joint_limits"],
                               velocity_constraints=robot_params["velocity_limits"],
                               rs=robot.rs,
                               query_states=y,
                               sdf=sdf,
                               robot=robot,
                               num_latent_gps=dof,
                               parameters=robot_params,
                               train_sigma=trainable_params["sigma_obs"],
                               no_frames_for_spheres=robot_params["no_frames_for_spheres"],
                               robot_name=robot_params["robot_name"],
                               epsilon=planner_params["epsilon"],
                               q_mu=q_mu,
                               whiten=False
                               )

    # DEBUGING CODE FOR VISUALIZING THE SPHERES

    # This part of the code is used to visualize the spheres in the scene
    # and behaves differently depending on the robot due to how its respective 
    # urdf file is structured. To have a complete tensorflow implementation, we are
    # transforming the relative positions of the spheres to absolute positions in the
    # world frame using the FK matrices. This is done in the sampler.py file at initialization
    # of the Sampler class and in its _fk_cost function. To introduce other robots, 
    # you need to follow the same file structure as data/robots/ and data/problemsets.
    # The main parts of interest are the config file and the urdf file. The config file
    # will have the joint limits, the number of spheres, the number of frames of reference
    # with respect to which the spheres are defined, the number of degrees of freedom of the
    # robot, and which frames of reference are needed to obtain the world frame position. 
    # The urdf file will contain the definition of the robot, the frames of reference
    # and relative sphere placement.

    # For visualization, similar to the joint positions debugging code, 
    # the location of the spheres is identified by a red line from the sphere world position 
    # to +0.05 in each cartesian direction. The sphere locations that you 
    # are currently debugging are colored green. 
    # To make them fit, you have to change their offset in the
    # initialization of the Sampler class in sampler.py.
    # Finally, make sure that the radius of the spheres is correctly defined and ordered in the config file.

    if graphics_params["debug_sphere_positions"]:
        start_pos = planner.likelihood.sampler._fk_cost(start_joints.reshape(dof, -1))

        for i, pos in enumerate(start_pos):
            aux_pos = np.array(pos).copy()
            aux_pos[0] += 0.1
            aux_pos[1] += 0.1
            aux_pos[2] += 0.1
            # if i >= 17 and i < 20:
            #     p.addUserDebugLine(pos, aux_pos, lineColorRGB=[0, 1, 0],
            #                     lineWidth=5.0, lifeTime=0, physicsClientId=env.sim.physicsClient)
            # else:
            p.addUserDebugLine(pos, aux_pos, lineColorRGB=[0, 1, 0],
                               lineWidth=5.0, lifeTime=0, physicsClientId=env.sim.physicsClient)

        base_pos, base_rot = p.getBasePositionAndOrientation(robot.robot_model)

        p.resetBasePositionAndOrientation(robot.robot_model, (base_pos[0], base_pos[1], base_pos[2]), base_rot)

        env.loop()

    # ENDING DEBUGING CODE FOR VISUALIZING THE SPHERES

    disable_param_opt(planner, trainable_params)
    robot.set_joint_position(start_joints)
    training_loop(model=planner, num_steps=num_steps, data=X)
    sample_mean, best_sample, samples, uncertainties = planner.sample_from_posterior(Xnew, robot)
    robot.set_joint_position(start_joints)
    tf.print(planner.likelihood.variance, summarize=-1)
    robot.enable_collision_active_links(-1)

    if graphics_params["visualize_best_sample"]:
        link_pos, _ = robot.compute_joint_positions(np.reshape(start_joints, (dof, 1)),
                                                    robot_params["craig_dh_convention"])
        prev = link_pos[-1]

        for joint_config in best_sample:
            link_pos, _ = robot.compute_joint_positions(np.reshape(joint_config, (dof, 1)),
                                                        robot_params["craig_dh_convention"])
            link_pos = np.array(link_pos[-1])
            p.addUserDebugLine(prev, link_pos, lineColorRGB=[0, 1, 0],
                               lineWidth=5.0, lifeTime=0, physicsClientId=env.sim.physicsClient)

            prev = link_pos

        link_pos, _ = robot.compute_joint_positions(np.reshape(end_joints, (dof, 1)),
                                                    robot_params["craig_dh_convention"])
        link_pos = np.array(link_pos[-1])
        p.addUserDebugLine(prev, link_pos, lineColorRGB=[0, 0, 1],
                           lineWidth=5.0, lifeTime=0, physicsClientId=env.sim.physicsClient)

    # PLOT THE MEAN OF THE SAMPLES AND THE UNCERTAINTY in the path of the robot END EFFECTOR
    if graphics_params["visualize_ee_path_uncertainty"]:
        link_pos, _ = robot.compute_joint_positions(np.reshape(start_joints, (dof, 1)),
                                                    robot_params["craig_dh_convention"])
        prev = link_pos[-1]

        t = np.linspace(0, 2 * np.pi, 50)
        cos = np.cos(t)
        sin = np.sin(t)
        for joint_config, unc in zip(sample_mean, uncertainties):
            link_pos, _ = robot.compute_joint_positions(np.reshape(joint_config, (dof, 1)),
                                                        robot_params["craig_dh_convention"])
            link_pos = np.array(link_pos[-1])
            p.addUserDebugLine(prev, link_pos, lineColorRGB=[0, 0, 1],
                               lineWidth=5.0, lifeTime=0, physicsClientId=env.sim.physicsClient)
            prev = link_pos
            rx, ry, rz = unc[0], unc[1], unc[2]
            prev_xx = [rx * cos[0], ry * sin[0], 0] + link_pos
            prev_yy = [rx * cos[0], 0, rz * sin[0]] + link_pos
            prev_zz = [0, ry * cos[0], rz * sin[0]] + link_pos
            for i in range(1, len(t)):
                xx = [rx * cos[i], ry * sin[i], 0] + link_pos
                yy = [rx * cos[i], 0, rz * sin[i]] + link_pos
                zz = [0, ry * cos[i], rz * sin[i]] + link_pos
                p.addUserDebugLine(prev_xx, xx, lineColorRGB=[1, 0, 0],
                                   lineWidth=5.0, lifeTime=0, physicsClientId=env.sim.physicsClient)
                p.addUserDebugLine(prev_yy, yy, lineColorRGB=[0, 1, 0],
                                   lineWidth=5.0, lifeTime=0, physicsClientId=env.sim.physicsClient)
                p.addUserDebugLine(prev_zz, zz, lineColorRGB=[0, 0, 1],
                                   lineWidth=5.0, lifeTime=0, physicsClientId=env.sim.physicsClient)
                prev_xx = xx
                prev_yy = yy
                prev_zz = zz

        link_pos, _ = robot.compute_joint_positions(np.reshape(end_joints, (dof, 1)),
                                                    robot_params["craig_dh_convention"])
        link_pos = np.array(link_pos[-1])
        p.addUserDebugLine(prev, link_pos, lineColorRGB=[0, 0, 1],
                           lineWidth=5.0, lifeTime=0, physicsClientId=env.sim.physicsClient)

    # PLOT THE SAMPLES
    if graphics_params["visualize_drawn_samples"]:
        for sample in samples:
            link_pos, _ = robot.compute_joint_positions(np.reshape(start_joints, (dof, 1)),
                                                        robot_params["craig_dh_convention"])
            prev = link_pos[-1]

            for joint_config in sample:
                link_pos, _ = robot.compute_joint_positions(np.reshape(joint_config, (dof, 1)),
                                                            robot_params["craig_dh_convention"])
                link_pos = np.array(link_pos[-1])
                p.addUserDebugLine(prev, link_pos, lineColorRGB=[1, 0, 0],
                                   lineWidth=5.0, lifeTime=0, physicsClientId=env.sim.physicsClient)

                prev = link_pos
            link_pos, _ = robot.compute_joint_positions(np.reshape(end_joints, (dof, 1)),
                                                        robot_params["craig_dh_convention"])
            link_pos = np.array(link_pos[-1])
            p.addUserDebugLine(prev, link_pos, lineColorRGB=[0, 0, 1],
                               lineWidth=5.0, lifeTime=0, physicsClientId=env.sim.physicsClient)

    print(f" alpha {planner.alpha}")
    for kern in planner.kernel.kernels:
        print(f" lengthscale {kern.lengthscales}")
        print(f" variance {kern.variance}")
    res = robot.move_to_ee_config(best_sample)
    return res


def disable_param_opt(planner, trainable_params):
    # Set priors to parameters here

    # planner.likelihood.variance.prior = tfp.distributions.Normal(planner.likelihood.variance,
    #                                                              gpflow.utilities.to_default_float(0.0001)

    planner.alpha.prior = tfp.distributions.Normal(planner.alpha,
                                                   gpflow.utilities.to_default_float(5))

    #

    for kern in planner.kernel.kernels:
        set_trainable(kern.variance, trainable_params["kernel_variance"])
        set_trainable(kern.lengthscales, trainable_params["lengthscales"])

    set_trainable(planner.inducing_variable, trainable_params["inducing_variable"])
    set_trainable(planner.alpha, trainable_params["alpha"])
    set_trainable(planner.likelihood.variance, trainable_params["sigma_obs"])
    set_trainable(planner._q_mu, trainable_params["q_mu"])
    set_trainable(planner._q_sqrt, trainable_params["q_sqrt"])


def draw_active_config(robot: object, config_array: np.ndarray, color: int, client: int) -> None:
    assert 0 <= color <= 2
    color_arr = [0, 0, 0]
    color_arr[color] += 1

    prev = robot.base_pose[:3, 3]
    print(prev)
    for joint in config_array:
        cur = joint
        p.addUserDebugLine(prev, cur, lineColorRGB=color_arr,
                           lineWidth=5.0, lifeTime=0, physicsClientId=client)
        prev = cur


def import_problemsets(robot_name):
    sys.path.insert(0, os.path.abspath('data/problemsets'))
    if robot_name == "franka":
        from franka import Problemset
    elif robot_name == "ur10":
        from ur10 import Problemset
    elif robot_name == "wam":
        from wam import Problemset
    elif robot_name == "kuka":
        from kuka import Problemset
    else:
        print("Robot not available. Check params file and try again... The simulator will now exit.")
        sys.exit(-1)
    return Problemset


def decay_sigma(sigma_obs, num_latent_gps, decay_rate):
    func = tf.range(num_latent_gps + 1)
    return tf.map_fn(lambda i: sigma_obs / (decay_rate * tf.cast(i + 1, dtype=default_float())), func,
                     fn_output_signature=default_float())


def get_root_package_path():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
