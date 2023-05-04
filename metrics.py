import numpy as np

def jerk(position, time):
    """
    Calculates the jerk of a trajectory. Returns the sum of the absolute values for each sphere on the EE.
    """
    velocity = np.array([np.gradient(position[..., i], time) for i in range(3)]).transpose()
    acceleration = np.array([np.gradient(velocity[..., i], time) for i in range(3)]).transpose()
    jerk = np.array([np.gradient(acceleration[..., i], time) for i in range(3)]).transpose()
    return np.sum(np.linalg.norm(jerk, axis=1))

def path_length(position):
    distance = np.linalg.norm(np.diff(position, axis=0), axis=1)
    return np.sum(distance)

def total_variation(traj, timesteps):
    """
    Calculates the total variation of a trajectory.
    :param traj: An array of values representing the trajectory. Shape: (N, 3)
    :param timesteps: An array of values representing the time steps. Shape: (N,)
    :return: The total variation of the trajectory.
    """
    diffs = np.diff(traj, axis=0)
    time_diffs = np.diff(timesteps)
    velocities = np.linalg.norm(diffs / time_diffs[:, np.newaxis], axis=1)
    return np.sum(velocities)

def average_metrics(metrics_dict):
    avg_dict = []
    for key in metrics_dict.keys():
        avg_dict.append(np.mean(metrics_dict[key]))

    return np.mean(avg_dict), np.sqrt(np.std(avg_dict)) * 2

degree_of_freedom = 3
end_time = 1
scale = 100
grid_spacing_Xnew = 150
# time = np.squeeze(np.array( [np.full(degree_of_freedom, i) for i in np.linspace(0, end_time * scale, grid_spacing_Xnew)], dtype=np.float64))
time = np.linspace(0, end_time * scale, grid_spacing_Xnew)
from collections import defaultdict
to_keep_length = defaultdict(list)
to_keep_jerk = defaultdict(list)
to_keep_tv = defaultdict(list)
robot = "franka"
problemset = "bookshelves"
print("Robot: {}, Problemset: {}".format(robot, problemset))
max_jerk = float('-inf')
gpmp2 = np.load('/home/freshpate/dump.npy')
print(gpmp2.shape)
import sys
# sys.exit()
for idx, traj in enumerate(gpmp2):
    jk = jerk(traj, time)
    pl = path_length(traj)
    tv = total_variation(traj, time)

    # print(see3, see4)
    # np.save('data/jerk_{}_{}.npy'.format(i, j), jerk)
    # np.save('data/path_length_{}_{}.npy'.format(i, j), path_length)
    # np.save('data/total_variation_{}_{}.npy'.format(i, j), total_variation)
    # keep metrics in the dictionary
    # max_jerk = max(max_jerk, np.max(np.abs(jerk)))

    to_keep_length[idx].append(pl)
    to_keep_jerk[idx].append(jk)
    to_keep_tv[idx].append(tv)

print("Average jerk: {}, std {}".format(*average_metrics(to_keep_jerk)))
print("Average path length: {}, std {}".format(*average_metrics(to_keep_length)))
print("Average total variation: {}, std {}".format(*average_metrics(to_keep_tv)))
sys.exit()
for i in range(1):
    for j in range(20):
        # Load the data
        try:
            position = np.load('/home/lucasc/git/vgpmp/data/saved_paths/{}/{}/pair_{}_run_{}.npy'.format(robot, problemset, j, i))
        except:
            continue
        # time = np.load('data/time_{}_{}.npy'.format(i, j))
        # Compute the metrics
        jk = jerk(position, time)
        pl = path_length(position)
        tv = total_variation(position, time)

        # print(see3, see4)
        # np.save('data/jerk_{}_{}.npy'.format(i, j), jerk)
        # np.save('data/path_length_{}_{}.npy'.format(i, j), path_length)
        # np.save('data/total_variation_{}_{}.npy'.format(i, j), total_variation)
        # keep metrics in the dictionary
        # max_jerk = max(max_jerk, np.max(np.abs(jerk)))

        to_keep_length[j].append(pl)
        to_keep_jerk[j].append(jk)
        to_keep_tv[i].append(tv)


print("Average jerk: {}, std {}".format(*average_metrics(to_keep_jerk)))
print("Average path length: {}, std {}".format(*average_metrics(to_keep_length)))
print("Average total variation: {}, std {}".format(*average_metrics(to_keep_tv)))