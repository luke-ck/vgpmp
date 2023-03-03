import numpy as np

def compute_jerk(position, time):
    velocity = np.array([np.gradient(position[..., i], time) for i in range(3)]).transpose()
    acceleration = np.array([np.gradient(velocity[..., i], time) for i in range(3)]).transpose()
    jerk = np.array([np.gradient(acceleration[..., i], time) for i in range(3)]).transpose()
    return np.sum(np.abs(jerk))

def compute_path_length(position):
    distances = np.sqrt(np.sum(np.diff(position, axis=0)**2, axis=1))
    path_length = np.sum(distances)
    return path_length

def compute_total_variation(position, time):
    velocity = np.array([np.gradient(position[..., i], time) for i in range(3)]).transpose()
    total_variation = np.sum(np.abs(np.array([np.gradient(velocity[..., i], time) for i in range(3)]).transpose()))
    return total_variation

def jerk_jk(position, time):
    velocity = np.diff(position, axis=0) / np.diff(time)[..., np.newaxis]
    acceleration = np.diff(velocity, axis=0) / np.diff(time[1:])[..., np.newaxis]
    jerk = np.diff(acceleration, axis=0) / np.diff(time[2:])[..., np.newaxis]
    return np.sum(np.abs(jerk))

def path_length_jk(position):
    distance = np.linalg.norm(np.diff(position, axis=0), axis=1)
    return np.sum(distance)

def total_variation_jk(position, time):
    velocity = np.diff(position, axis=0) / np.diff(time)[..., np.newaxis]
    return np.sum(np.abs(np.diff(velocity, axis=0)))

def total_variation_jj(traj, timesteps):
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
max_jerk = float('-inf')
for i in range(5):
    for j in range(55):
        # Load the data
        try:
            position = np.load('/home/freshpate/saved_paths/{}/{}/pair_{}_run_{}.npy'.format(robot, problemset, j, i))
        except:
            continue
        # time = np.load('data/time_{}_{}.npy'.format(i, j))
        # Compute the metrics
        jerk = compute_jerk(position, time)
        path_length = compute_path_length(position)
        total_variation = compute_total_variation(position, time)
        # Save the metrics
        see = jerk_jk(position, time)
        see2 = path_length_jk(position)
        see3 = total_variation_jj(position, time)
        see4 = total_variation_jk(position, time)
        print(see3, see4)
        # np.save('data/jerk_{}_{}.npy'.format(i, j), jerk)
        # np.save('data/path_length_{}_{}.npy'.format(i, j), path_length)
        # np.save('data/total_variation_{}_{}.npy'.format(i, j), total_variation)
        # keep metrics in the dictionary
        max_jerk = max(max_jerk, np.max(np.abs(jerk)))
        to_keep_length[j].append(path_length)
        to_keep_jerk[j].append(jerk)
        to_keep_tv[i].append(total_variation)

# print(to_keep_tv)
# print(max_jerk)
avg_tv = defaultdict(float)
std_tv = defaultdict(float)
for key in to_keep_tv.keys():
    avg_tv[key] = np.mean(to_keep_tv[key])
    std_tv[key] = np.std(to_keep_tv[key])
print(avg_tv)
print(std_tv)
