# A Unifying Framework for Variational Gaussian Process Motion Planning

This repo contains the code used to generate the results of our paper, A Unifying Framework for Variational Gaussian Process Motion Planning, which was accepted at AISTATS 2024 and can be found here: https://arxiv.org/abs/2309.00854


### Installation Guide

We recommend running the code on a Linux machine. First, go to the base directory and create a Python3 environment with the strict requirement of python <= 3.9. For example, with conda do the following:
```bash
conda create -n vgpmp python=3.9
conda activate vgpmp
```
Then
```bash
pip install -e requirements.txt
```

After this is done you need to clone the following repo: https://github.com/j-wilson/GPflowSampling. Install their library which enables pathwise conditioning for GPs in TensorFlow.
Additionally, you need to install pyassimp which depends on your OS. For example, on Manjaro or any Arch based distribution you can simply do:
```bash
yay -S python-pyassimp
```
For Ubuntu, do:
```bash
sudo apt-get install python-pyassimp
```
or
```bash
sudo apt-get install python3-pyassimp
```

### How to use the repo

Now you are ready to perform experiments. First run `export TF_CPP_MIN_LOG_LEVEL=2` in the terminal for your sanity. Then, simply run 
```bash
python benchmarking.py
```
in a terminal. By default, it should load the WAM arm in the industrial environment. 

If you want to change the position of the environment, go to the `parameters.yaml` file on line 5. That list contains the x,y,z coordinates and below it you can see the orientation given in quaternion notation.

##### Changing environment

We only support two environments for now, industrial and bookshelves. To use the industrial environment, your `parameters.yaml` file should have the following information for lines 7-9 and 34:
```yaml
environment_name: "industrial"
environment_file_name: "industrial"
sdf_file_name: "industrial_vgpmp"
problemset_name: "industrial"
```
The environment positions you should use for our problem sets are:

Franka -> [-0.2, 0, -0.2]

Kuka -> [-0.2, 0, -0.2]

WAM -> [-0.2, 0.0, 0.08]

UR10 -> [-0.2, 0.0, 0.08]

For the bookshelves environment, use the following:
```yaml
environment_name: "bookshelves"
environment_file_name: "bookshelves_mesh"
sdf_file_name: "bookshelves_center_vgpmp"
problemset_name: "bookshelves"
```
The environment positions you should use for our problem sets are:

Franka -> [ 0.62, -0.15, 0.834] 

Kuka -> [ 0.62, -0.15, 0.834] 

WAM -> [ 0.85, -0.15, 0.834]

UR10 -> [ 0.95, -0.15, 0.834]

For all environments, use the orientation [0, 0, 0, 1]. SDF rotation is not yet implemented. Therefore, changing the orientation here will only affect the .obj file's orientation, not the .sdf's. As a result, the algorithm will not function correctly.


##### Checking robot properties

If you go to `data->problemsets->replace-robot-name-here.py`, you will see robots available for use. As you saw above, we support Franka, Kuka iiwa7, WAM and UR10.
In their respective files, for example in franka.py, you can change multiple things:
- The states/problemsets define the motion planning problems that need to be solved. Essentially, you specify joint angles that are within the robot's limits, and during deployment the paths are all combinations of 2.
- pos_and_orn defines the position (x, y, z) and orientation (quaternion) of the robot. Here you can use the orientation as well. herefore, if rotation is required, adjust the robot's orientation rather than the environment's, for the time being.
- planner_params defines all planner parameters for optimization, here you just specify their values, such as kernel lengthscales, variances, $\sigma_{obs}$, number of inducing variables, etc. For the mean of the inducing variables, $q_{\mu}$, change line 168 in gpflow_vgpmp/utils/miscellaneous.py. There are 3 methods implemented: `None` (mean initialized as middle of joint constraints), `linear` (linear interpolation between start and end configuration), `waypoint` (3 points, start, start / 2 + end / 2, end).
- To set which variables to train and which are fixed during training, go to `parameters.yaml` and change the `trainable_params` between `True` and `False` at your leisure.

##### Path visualization 

In `parameters.yaml`, you can change some `graphics` attributes to visualize different outputs from our algorithm. The main options for importance are:
```yaml
visualize_best_sample: True
visualize_ee_path_uncertainty: False
visualize_drawn_samples: False
```
`visualize_best_sample` shows the most collision free sample, `visualize_ee_path_uncertainty` shows the motion plan uncertainty from the GP, `visualize_drawn_samples` shows some samples that are drawn from the GP.



