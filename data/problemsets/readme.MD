Any **environment dependent** information should be stored in the `data/problemsets` directory. This includes the following:

* _planner parameters_: `sigma_obs`, `epsilon`, the kernel `variance` and `lengthscale`, and `alpha`
* _robot configuration_: this includes the default pose of the robot in the given environment, the joint names and the robot position in world coordinates
* _the problemset_: the set of motion planning problems to be solved in the given environment. During runtime, we take every combination of start and goal configurations given this set.

Each problemset is robot (and environment) dependent. Each robot problemset is an instance of the `Problemset` class. As such, users wanting to use their own robot should create a new class which inherits from `Problemset` and implements all the abstract methods to define the information outlined above. Finally, to use the custom problemset one should add the relevant import in the `import_problemsets` function found in `gpflow_vgpmp/utils/miscellaneous.py`.