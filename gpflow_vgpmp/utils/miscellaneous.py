import itertools
import os
import sys
import time
from contextlib import contextmanager

import numpy as np
import pybullet as p
import tensorflow as tf
from tqdm import tqdm

import gpflow_vgpmp.utils.problemsets as problemsets

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except RuntimeError:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


def timing(f):
    def wrap(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        end = time.time()
        print('{:s} function took {:.3f} ms'.format(
            f.__name__, (end - start) * 1000.0))

        return ret

    return wrap


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


def plot(EE, title):
    def on_close(event):
        print('Closed Figure!')

    import matplotlib.pyplot as plt
    plt.rc('figure', dpi=256)
    plt.rc('font', family='serif', size=12)
    plt.rc('text', usetex=False)
    start = [[79, 232, 222]]  # turqoise
    end = [[255, 255, 102]]  # yellowish
    middle = [[50, 124, 64]] * (len(EE) - 2)

    c = np.array([start + middle + end]) / 255
    c = c.reshape((len(EE), 3))

    X = EE[:, 0]
    Y = EE[:, 1]
    Z = EE[:, 2]

    alphas = np.array([1] + [0.5] * (len(EE) - 2) + [1])

    ax = plt.axes(projection='3d')

    ax.plot3D(X, Y, Z, 'gray')
    start = ax.scatter3D(X[0], Y[0], Z[0], c=c[0], alpha=alphas[0], marker='o')
    ax.scatter3D(X[1:-1], Y[1:-1], Z[1:-1], c=c[1:-1], marker='^')
    goal = ax.scatter3D(X[-1], Y[-1], Z[-1], c=c[-1], alpha=alphas[-1], marker='o')
    ax.set_title("VGPMP Trajectory for PR2 on Bookshelves Dataset")
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    ax.legend([start, goal], ["Start State", "Goal State"], fontsize=8)
    # rotate the axes and update
    for angle in range(0, 360):
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(.001)

    timestr = time.strftime("%Y-%m-%dT%H:%M:%S")
    dir = os.getcwd() + '/data/plots/'
    plt.savefig(dir + title + "-" + timestr)


def write_parameter_to_file(model_parameter, filepath):
    param = tf.strings.format("{}\n", model_parameter, summarize=-1)
    tf.io.write_file(f"{filepath}.txt", param)


def print_state(query_states, right_arm, robot) -> None:
    print("start configuration: ", query_states[0])
    print("goal configuration: ", query_states[1])

    base = robot.get_base_pos()

    print("base of robot:", base)
    print("FK arm base: ", right_arm.base_pose)
    print("arm base as computed by pybullet: ", right_arm.base_pose_cart)


def create_problems(problemset):
    r"""
    Open the problemsets
    """
    # Start and end joint angles
    n_states, states = problemsets.states(problemset)
    print('There are %s total robot positions' % (n_states))
    # all possible combinations of 2 pairs
    benchmark = list(itertools.combinations(states, 2))
    print('And a total of %d problems in the %s problemset' %
          (len(benchmark), problemset))
    names = problemsets.joint_names(problemset)
    pose = problemsets.default_base_pose(problemset)
    joint_values = problemsets.default_joint_values(problemset)
    active_joints = problemsets.active_joints(problemset)
    sphere_links = problemsets.active_sphere_links()
    return benchmark, names, pose, joint_values, active_joints, sphere_links


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


def init_trainset(grid_spacing, degree_of_freedom, start_joints, end_joints):
    X = tf.convert_to_tensor(np.array(
        [np.full(degree_of_freedom, i) for i in np.linspace(0, 1, grid_spacing)], dtype=np.float64))
    y = tf.concat([start_joints.reshape((1, degree_of_freedom)), end_joints.reshape((1, degree_of_freedom))], axis=0)
    Xnew = tf.convert_to_tensor(np.array(
        [np.full(degree_of_freedom, i) for i in np.linspace(0, 1, grid_spacing // 2)], dtype=np.float64))
    return X, y, Xnew


def solve_planning_problem(start_joints, end_joints, params):
    grid_spacing = params["time_spacing"]
    dof = params["dof"]
    max_time = params["max_time"]
    num_steps = params["num_steps"]
    X, y, Xnew = init_trainset(grid_spacing, dof, start_joints, end_joints)
    num_data, num_output_dims = y.shape
    planner = VGPMP.initialize(num_data=num_data,
                               num_output_dims=num_output_dims,
                               num_spheres=planner_params["num_spheres"],
                               num_inducing=planner_params["num_inducing"],
                               sigma_obs=planner_params["sigma_obs"],
                               alpha=planner_params["alpha"],
                               learning_rate=planner_params["learning_rate"],
                               rs=robot.rs,
                               query_states=y,
                               sdf=sdf,
                               robot=robot,
                               mean_function='default',
                               num_latent_gps=dof,
                               parameters=params.robot
                               )
    disable_hyperparam_opt(planner)
    training_loop(planner, num_steps, data=X, max_time=max_time, optimizer=planner.optimizer)
    joint_configs = planner.sample_from_posterior(Xnew)

    robot.enable_collision_active_links(-1)
    robot.set_joint_position(start_joints)
    _joint_pos = robot.compute_joint_positions(start_joints.reshape((dof, 1)))
    link_joint_pos = _joint_pos + robot.joint_link_offsets

    EE = [link_joint_pos[-1]]
    prev = link_joint_pos[-1]
    for joint_config in joint_configs:
        joint_pos = robot.compute_joint_positions(joint_config.reshape((dof, 1)))
        joint_pos += robot.joint_link_offsets
        joint_pos = np.array(link_pos[-1])
        p.addUserDebugLine(prev, link_pos, lineColorRGB=[0, 0, 1],
                           lineWidth=5.0, lifeTime=0, physicsClientId=env.sim.physicsClient)
        time.sleep(2)
        prev = link_pos
        EE.append(link_pos)

    EE = np.array(EE)

    res = robot.move_to_ee_config(joint_configs)
    return res


def disable_hyperparam_opt(planner):
    set_trainable(planner.kernel.kernels, False)
    set_trainable(planner.inducing_variable, False)


def draw_active_config(robot: object, config_array: np.ndarray, color: int, client: int) -> None:
    assert 0 <= color <= 2
    color_arr = [0, 0, 0]
    color_arr[color] += 1

    prev = robot.get_base_pos().reshape(3)
    for joint in config_array:
        cur = joint
        p.addUserDebugLine(prev, cur, lineColorRGB=color_arr,
                           lineWidth=5.0, lifeTime=0, physicsClientId=client)
        prev = cur


def benchmarking():
    env = RobotSimulator()
    params = env.get_simulation_params()
    planner_params = params.planner

    robot = env.robot
    scene = env.scene
    sdf = env.sdf
    scene = params.scene["problemset"]
    queries, initial_config_names, initial_config_pose, initial_config_joints, active_joints, sphere_links = create_problems(
        scene)

    total_problems = len(queries)

    for query in queries:
        start_joints = np.array(query[0], dtype=np.float64)
        end_joints = np.array(query[1], dtype=np.float64)
        robot.initialise(start_joints, active_joints, sphere_links, initial_config_names, initial_config_pose,
                         initial_config_joints)
        dof = robot.dof
        planner_params["dof"] = dof
        res = solve_planning_problem(start_joints, end_joints, planner_params)
        if res:
            solved_problems += 1
    print(f"For scene {scene} planner solved {solved_problems} out of {total_problems} problems")
