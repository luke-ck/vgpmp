import importlib
import itertools
import os
import signal
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


def are_all_elements_integers(tup):
    return all(isinstance(elem, int) for elem in tup)


def optimization_step(model, closure, optimizer):
    r"""

    """
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        loss = closure()

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


def shutdown():
    p.disconnect()
    print("Shutting down the simulator")
    sys.exit(0)


def training_loop(model, data, num_steps):
    r"""

    """
    print("Starting training....")
    # data = tf.broadcast_to(tf.sort(tf.random.uniform((data.shape[0], 1), 0.0, 1.0, dtype=tf.float64), axis=0), [data.shape[0], 7])
    tf_optimization_step = tf.function(optimization_step)
    closure = model.training_loss_closure(data)
    step_iterator = tqdm(range(num_steps))

    for _ in step_iterator:
        loss = tf_optimization_step(model, closure, model.optimizer)
        step_iterator.set_postfix_str(f"ELBO: {-loss:.3e}")

    # print model parameters
    # for kern in model.kernel.kernels:
    #     tf.print(f"model lengthscale: {kern.lengthscales} \nmodel variance: {kern.variance}")
    #
    # tf.print(f"model noise variance: {model.likelihood.variance}")
    # tf.print(f"model q_mu: {model.q_mu}")
    # tf.print(f"model q_sqrt: {model.q_sqrt}", summarize=-1)


def init_trainset(grid_spacing_X, grid_spacing_Xnew, degree_of_freedom, start_joints, end_joints, scale=100,
                  end_time=1):
    X = tf.convert_to_tensor(np.array(
        [np.full(degree_of_freedom, i) for i in np.linspace(0, end_time * scale, grid_spacing_X)], dtype=np.float64))

    y = tf.concat([start_joints.reshape((1, degree_of_freedom)), end_joints.reshape((1, degree_of_freedom))], axis=0)
    Xnew = tf.convert_to_tensor(np.array(
        [np.full(degree_of_freedom, i) for i in np.linspace(0, end_time * scale, grid_spacing_Xnew)], dtype=np.float64))
    return X, y, Xnew


def detect_joint_limit_proximity(limits, q_mu):
    """
    if trajectory is too close to joint limits, return True
    limits has shape (2, dof) where first row is upper limit and second row is lower limit
    """
    if q_mu is None:
        return False
    else:
        return np.any(np.abs(q_mu - limits[:, 0]) < 0.1) or np.any(np.abs(q_mu - limits[:, 1]) < 0.1)


def solve_planning_problem(env, start_joints, end_joints, run=0, k=0):
    from gpflow_vgpmp.models.vgpmp import VGPMP

    planner_params = env.config["planner_params"]
    graphics_params = env.config["graphics_params"]
    trainable_params = env.config["trainable_params"]

    time_spacing_x = planner_params["time_spacing_X"]
    time_spacing_xnew = planner_params["time_spacing_Xnew"]
    num_steps = planner_params["num_steps"]

    dof = env.robot.dof

    X, y, Xnew = init_trainset(time_spacing_x, time_spacing_xnew, dof, start_joints, end_joints, scale=1)
    num_data, num_output_dims = y.shape

    # q_mu = np.array(robot_params["q_mu"], dtype=np.float64).reshape(1, dof) if robot_params["q_mu"] != "None" else None
    q_mu = np.array([y[0] + (y[1] - y[0]) * i / (planner_params["num_inducing"]) for i in
                     range(planner_params["num_inducing"])])  # all ish

    planner = VGPMP.initialize(sdf=env.sdf,
                               robot=env.robot,
                               sampler=env.sampler,
                               query_states=y,
                               scene_offset=env.scene.position,
                               q_mu=q_mu,
                               **env.config['planner_params'])

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
        start_pos = planner.likelihood.sampler.forward_kinematics_cost(start_joints.reshape(dof, -1))

        for i, pos in enumerate(start_pos):
            aux_pos = np.array(pos).copy()
            aux_pos[0] += 0.1
            aux_pos[1] += 0.1
            aux_pos[2] += 0.1
            # if i >= 17 and i < 20:
            #     p.addUserDebugLine(pos, aux_pos, lineColorRGB=[0, 1, 0],
            #                     lineWidth=5.0, lifeTime=0, physicsClientId=env.sim.client)
            # else:
            p.addUserDebugLine(pos, aux_pos, lineColorRGB=[0, 1, 0],
                               lineWidth=5.0, lifeTime=0, physicsClientId=env.client)

        # base_pos, base_rot = p.getBasePositionAndOrientation(env.robot.robot_model)
        #
        # p.resetBasePositionAndOrientation(env.robot.robot_model, (base_pos[0], base_pos[1], base_pos[2]), base_rot)

        env.loop()

    # ENDING DEBUGING CODE FOR VISUALIZING THE SPHERES

    disable_param_opt(planner, trainable_params)

    env.robot.set_current_joint_config(np.squeeze(start_joints))
    training_loop(model=planner, num_steps=num_steps, data=X)
    sample_mean, best_sample, samples, uncertainties = planner.sample_from_posterior(Xnew, env.robot, graphics_params[
        "visualize_ee_path_uncertainty"])
    env.robot.set_current_joint_config(np.squeeze(start_joints))
    env.robot.enable_collision_active_links(-1)

    # SAVE THE BEST SAMPLE

    if graphics_params["visualize_best_sample"]:
        link_pos, _ = env.robot.compute_joint_positions(np.reshape(start_joints, (dof, 1)))
        prev = link_pos[-1]

        for joint_config in best_sample:
            link_pos, _ = env.robot.compute_joint_positions(np.reshape(joint_config, (dof, 1)))
            link_pos = np.array(link_pos[-1])
            p.addUserDebugLine(prev, link_pos, lineColorRGB=[0, 1, 0],
                               lineWidth=5.0, lifeTime=0, physicsClientId=env.client)

            prev = link_pos

        link_pos, _ = env.robot.compute_joint_positions(np.reshape(end_joints, (dof, 1)))
        link_pos = np.array(link_pos[-1])
        p.addUserDebugLine(prev, link_pos, lineColorRGB=[0, 0, 1],
                           lineWidth=5.0, lifeTime=0, physicsClientId=env.client)

    # # PLOT THE MEAN OF THE SAMPLES AND THE UNCERTAINTY in the path of the robot END EFFECTOR
    if graphics_params["visualize_ee_path_uncertainty"]:
        link_pos, _ = env.robot.compute_joint_positions(np.reshape(start_joints, (dof, 1)))
        prev = link_pos[-1]

        t = np.linspace(0, 2 * np.pi, 50)
        cos = np.cos(t)
        sin = np.sin(t)
        for joint_config, unc in zip(sample_mean, uncertainties):
            link_pos, _ = env.robot.compute_joint_positions(np.reshape(joint_config, (dof, 1)))
            link_pos = np.array(link_pos[-1])
            p.addUserDebugLine(prev, link_pos, lineColorRGB=[0, 0, 1],
                               lineWidth=5.0, lifeTime=0, physicsClientId=env.client)
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
                                   lineWidth=5.0, lifeTime=0, physicsClientId=env.client)
                p.addUserDebugLine(prev_yy, yy, lineColorRGB=[0, 1, 0],
                                   lineWidth=5.0, lifeTime=0, physicsClientId=env.client)
                p.addUserDebugLine(prev_zz, zz, lineColorRGB=[0, 0, 1],
                                   lineWidth=5.0, lifeTime=0, physicsClientId=env.client)
                prev_xx = xx
                prev_yy = yy
                prev_zz = zz

        link_pos, _ = env.robot.compute_joint_positions(np.reshape(end_joints, (dof, 1)))
        link_pos = np.array(link_pos[-1])
        p.addUserDebugLine(prev, link_pos, lineColorRGB=[0, 0, 1],
                           lineWidth=5.0, lifeTime=0, physicsClientId=env.client)

    # # PLOT THE SAMPLES
    if graphics_params["visualize_drawn_samples"]:
        for sample in samples:
            link_pos, _ = env.robot.compute_joint_positions(np.reshape(start_joints, (dof, 1)))
            prev = link_pos[-1]

            for joint_config in sample:
                link_pos, _ = env.robot.compute_joint_positions(np.reshape(joint_config, (dof, 1)))
                link_pos = np.array(link_pos[-1])
                p.addUserDebugLine(prev, link_pos, lineColorRGB=[1, 0, 0],
                                   lineWidth=5.0, lifeTime=0, physicsClientId=env.client)

                prev = link_pos
            link_pos, _ = env.robot.compute_joint_positions(np.reshape(end_joints, (dof, 1)))
            link_pos = np.array(link_pos[-1])
            p.addUserDebugLine(prev, link_pos, lineColorRGB=[0, 0, 1],
                               lineWidth=5.0, lifeTime=0, physicsClientId=env.client)
    # move robot at each inducing point in q_mu
    # link_pos, _ = env.robot.compute_joint_positions(np.reshape(start_joints, (dof, 1)))
    # prev = link_pos[-1]
    # for i in range(len(planner.q_mu)):
    #     joint_config = planner.likelihood.joint_sigmoid(planner.q_mu[i])
    #     env.robot.set_current_joint_config(joint_config)
    #     # compute the end effector position
    #     link_pos, _ = env.robot.compute_joint_positions(np.reshape(joint_config, (dof, 1)))
    #     link_pos = np.array(link_pos[-1])
    #     p.addUserDebugLine(prev, link_pos, lineColorRGB=[1, 0, 0],
    #                         lineWidth=5.0, lifeTime=0, physicsClientId=env.client)
    #     prev = link_pos
    #     time.sleep(1)
    # print(f" alpha {planner.alpha}")
    # for kern in planner.kernel.kernels:
    #     print(f" lengthscale {kern.lengthscales}")
    #     print(f" variance {kern.variance}")
    # print(f" likelihood variance {planner.likelihood.variance}")
    res = env.robot.move_to_ee_config(best_sample)
    # pos = tf.vectorized_map(planner.likelihood.compute_fk_joints, best_sample)
    return res, best_sample


def disable_param_opt(planner, trainable_params):
    # Set priors to parameters here

    planner.likelihood.variance.prior = tfp.distributions.Normal(planner.likelihood.variance,
                                                                 gpflow.utilities.to_default_float(0.0001))

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


def decay_sigma(sigma_obs, num_latent_gps, decay_rate):
    func = tf.range(num_latent_gps + 1)
    return tf.map_fn(lambda i: sigma_obs / (decay_rate * tf.cast(i + 1, dtype=default_float())), func,
                     fn_output_signature=default_float())


def get_root_package_path():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
