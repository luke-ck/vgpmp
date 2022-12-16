import itertools
import os
import sys
import time
from contextlib import contextmanager
from sympy import lambdify, symbols, init_printing, Matrix, eye, sin, cos, pi
from gpflow.config import default_float
import numpy as np
import pybullet as p
import tensorflow as tf
import roboticstoolbox as rtb
import tensorflow_probability as tfp
import gpflow
from gpflow import set_trainable
from tqdm import tqdm

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
    Open the problemsets
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
    pose = Problemset.default_base_pose(problemset)
    joint_values = Problemset.default_joint_values(problemset)
    active_joints = Problemset.active_joints(problemset)
    return states, names, pose, joint_values, active_joints


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


def training_loop(model, data, num_steps, optimizer):
    r"""

    """
    print("Starting training....")
    # data = tf.broadcast_to(tf.sort(tf.random.uniform((10, 1), 0.0001, 0.9999, dtype=tf.float64), axis=0), [10, 7])
    tf_optimization_step = tf.function(optimization_step)
    closure = model.training_loss_closure(data)
    timeout_start = time.time()
    step_iterator = tqdm(range(num_steps))
    for step in step_iterator:
        loss = tf_optimization_step(model, closure, optimizer)
        step_iterator.set_postfix_str(f"ELBO: {-loss:.3e}")


def init_trainset(grid_spacing_X, grid_spacing_Xnew, degree_of_freedom, start_joints, end_joints, scale=100, end_time=1):
    X = tf.convert_to_tensor(np.array(
        [np.full(degree_of_freedom, i) for i in np.linspace(0, end_time * scale, grid_spacing_X)], dtype=np.float64))
    y = tf.concat([start_joints.reshape((1, degree_of_freedom)), end_joints.reshape((1, degree_of_freedom))], axis=0)
    Xnew = tf.convert_to_tensor(np.array(
        [np.full(degree_of_freedom, i) for i in np.linspace(0, end_time * scale, grid_spacing_Xnew)], dtype=np.float64))
    return X, y, Xnew


def solve_planning_problem(env, robot, sdf, start_joints, end_joints, robot_params, planner_params, scene_params, trainable_params):
    grid_spacing_X = planner_params["time_spacing_X"]
    grid_spacing_Xnew = planner_params["time_spacing_Xnew"]
    dof = robot_params["dof"]
    num_steps = planner_params["num_steps"]
    X, y, Xnew = init_trainset(grid_spacing_X, grid_spacing_Xnew, dof, start_joints, end_joints)
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
                               parameters=robot_params,
                               train_sigma=trainable_params["sigma_obs"],
                               q_mu=None,
                               whiten=False,
                               )

    disable_param_opt(planner, trainable_params)
    training_loop(model=planner, num_steps=num_steps, data=X, optimizer=planner.optimizer)
    sample_mean, best_sample, samples = planner.sample_from_posterior(Xnew)
    tf.print(planner.likelihood.variance, summarize=-1)
    robot.enable_collision_active_links(-1)
    robot.set_joint_position(start_joints)
    link_pos, _ = robot.compute_joint_positions(np.reshape(start_joints, (dof, 1)))
    prev = link_pos[-1]

    for joint_config in best_sample:
        link_pos, _ = robot.compute_joint_positions(np.reshape(joint_config, (dof, 1)))
        link_pos = np.array(link_pos[-1])
        p.addUserDebugLine(prev, link_pos, lineColorRGB=[0, 1, 0],
                           lineWidth=5.0, lifeTime=0, physicsClientId=env.sim.physicsClient)

        prev = link_pos

    # link_pos, _ = robot.compute_joint_positions(np.reshape(start_joints, (dof, 1)))
    # prev = link_pos[-1]

    # for joint_config in sample_mean:
    #     link_pos, _ = robot.compute_joint_positions(np.reshape(joint_config, (dof, 1)))
    #     link_pos = np.array(link_pos[-1])
    #     p.addUserDebugLine(prev, link_pos, lineColorRGB=[0, 0, 1],
    #                        lineWidth=5.0, lifeTime=0, physicsClientId=env.sim.physicsClient)

    #     p.addUserDebugLine(prev, link_pos, lineColorRGB=[1, 0, 0],
    #                        lineWidth=5.0, lifeTime=0, physicsClientId=env.sim.physicsClient)

    #     # time.sleep(0.05)
    #     prev = link_pos

    # for sample in samples:
    #     link_pos, _ = robot.compute_joint_positions(np.reshape(start_joints, (dof, 1)))
    #     prev = link_pos[-1]

    #     for joint_config in sample:
    #         link_pos, _ = robot.compute_joint_positions(np.reshape(joint_config, (dof, 1)))
    #         link_pos = np.array(link_pos[-1])
    #         p.addUserDebugLine(prev, link_pos, lineColorRGB=[1, 0, 0],
    #                         lineWidth=5.0, lifeTime=0, physicsClientId=env.sim.physicsClient)

    #         prev = link_pos
    link_pos, _ = robot.compute_joint_positions(np.reshape(end_joints, (dof, 1)))
    link_pos = np.array(link_pos[-1])
    p.addUserDebugLine(prev, link_pos, lineColorRGB=[0, 0, 1],
                       lineWidth=5.0, lifeTime=0, physicsClientId=env.sim.physicsClient)
    print(f" alpha {planner.alpha}")
    for kern in planner.kernel.kernels:
        print(f" lengthscale {kern.lengthscales}")
        print(f" variance {kern.variance}")
    res = robot.move_to_ee_config(best_sample, y[1])
    return res

def disable_param_opt(planner, trainable_params):

    # Set priors to parameters here

    # planner.likelihood.variance.prior = tfp.distributions.Normal(gpflow.utilities.to_default_float(0.0005),
    #                                                              gpflow.utilities.to_default_float(0.005))
    planner.alpha.prior = tfp.distributions.Normal(gpflow.utilities.to_default_float(10),
                                                                 gpflow.utilities.to_default_float(0.2))

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
        return Problemset
    elif robot_name == "pr2":
        from pr2 import Problemset
        return Problemset
    else:
        print("Robot not available. Check params file and try again... The simulator will now exit.")
        sys.exit(-1)

