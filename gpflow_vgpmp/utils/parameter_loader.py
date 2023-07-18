import itertools
import sys
from pathlib import Path
from typing import Tuple, Optional
import yaml
from gpflow_vgpmp.utils.miscellaneous import get_root_package_path
import copy

def load_yaml_config(scene_config):
    with open(scene_config, 'r') as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.constructor.ConstructorError as e:
            print(e)
    return config_dict


class ParameterLoader:
    """
    Load and extract data from a parameter files, which are used to configure the simulator. On start up, the loader
    configures the following parameters:
    - self.scene_params = parameters dealing with the scene (managing objects, sdf, the environment)
    - self.robot_params = parameters necessary to set up the robot (using robot specific config
    parameter file + general robot parameters found in the central config file)
    - self.graphic_params = visual parameters passed to SimulationThread to establish the pybullet connection
    - self.trainable_params = parameters that are used to configure the GP model (whether to train q_mu, q_sqrt, etc.)
    """

    def __init__(self):

        self.is_initialized = False
        self._params = None
        self.trainable_params = None
        self.planner_params = None
        self.graphics_params = None
        self.robot_params = None
        self.scene_params = None
        root_package_path = Path(get_root_package_path())
        self.root_path = root_package_path
        self.data_dir_path = root_package_path / 'data'

    @property
    def params(self) -> dict:
        assert self._params is not None, "Parameter Loader must be initialized before it can be accessed"
        return self._params

    def initialize(self, file_path: Path = None, params: Optional[dict] = None):
        """
        Initialize the parameter loader by loading the parameter file and extracting the data
        or by directly passing the params dict (this is used for testing)

        :param file_path: path to the parameter file
        :param params: dict containing the parameters
        """
        if file_path is not None:
            self.load_parameter_file(file_path)
        else:
            assert params is not None, "Either parameter_file_path or params must be specified"
            self._params = self.set_params(params)

    def set_params(self, params):
        """ Load and extract data from parameter file """

        robot_params, scene_params, trainable_params, graphic_params = params

        self.scene_params = scene_params["scene"]
        self.robot_params = robot_params["robot"]
        self.trainable_params = trainable_params["trainable_params"]
        self.graphics_params = graphic_params["graphics"]

        self.get_robot_config(self.robot_params)
        self.get_scene_config(self.scene_params)

        self.is_initialized = True
        return {
            'robot_params': self.robot_params,
            'scene_params': self.scene_params,
            'planner_params': self.planner_params,
            'trainable_params': self.trainable_params,
            'graphics_params': self.graphics_params
        }

    def get_robot_config(self, robot_params: dict):
        """ Load robot configuration from parameter file which is found together with the rest of the robot files
            The robot configuration is saved in the robot_params dict.
            The general idea is this: general robot parameters are saved in the main parameter file,
            and robot specific parameters (FK related, joint limits, spheres and so on)
            are saved in the robot specific parameter file.
         """
        robot_name = robot_params["robot_name"]  # specify which robot to load
        robots_data_dir_path = self.data_dir_path / "robots"
        robot_path = robots_data_dir_path / robot_name
        robot_config = robot_path / "config.yaml"
        config_dict = load_yaml_config(robot_config)
        config_dict["urdf_path"] = robot_path / config_dict["path"]

        # concatenate the robot parameters with the robot config
        # if there are any duplicates, robot parameters have precedence
        self.robot_params = {**config_dict, **robot_params}

    def get_scene_config(self, scene_params: dict):
        """Load scene paths require parameter_file_path for SDF and URDF files """

        environment_name = scene_params["environment_name"]
        environment_file_name = scene_params["environment_file_name"]
        sdf_file_name = scene_params["sdf_file_name"]
        scene_params["objects_path"] = []

        if scene_params["objects"] is not None and scene_params["objects"] is not []:
            objects_data_dir_path = self.data_dir_path / "objects"
            for obj in scene_params["objects"]:
                object_path = objects_data_dir_path / obj
                assert object_path.exists()
                scene_params["objects_path"].append(object_path)

        assert scene_params["benchmark"] is not None, "Benchmark attribute is not specified"
        assert type(scene_params["benchmark"]) is bool, "Benchmark attribute must be a boolean"
        if scene_params["benchmark"] is False:
            non_benchmark_attrs = scene_params["non_benchmark_attributes"]
            states = non_benchmark_attrs["states"]
            n_states = len(states)
            planner_params = non_benchmark_attrs["planner_params"]
            robot_pos_and_orn = tuple(non_benchmark_attrs["robot_pos_and_orn"])

        else:
            from data.problemsets.problemset import import_problemset

            robot_name = self.robot_params['robot_name']
            problemset_name = self.scene_params["benchmark_attributes"]["problemset_name"]
            # Start and end joint angles
            problemset = import_problemset(robot_name)
            n_states, states = problemset.states(problemset_name)

            planner_params = problemset.planner_params(problemset_name)
            robot_pos_and_orn = problemset.pos_and_orn(problemset_name)

        # all possible combinations of 2 pairs
        queries = list(itertools.combinations(states, 2))
        print(f'There are {n_states} total robot positions and a total of {len(queries)} problems')

        scene_params["queries"] = queries
        scene_params["robot_pos_and_orn"] = robot_pos_and_orn
        environment_path, sdf_path = self.get_assets_path(environment_name, environment_file_name, sdf_file_name)

        scene_params["sdf_path"] = sdf_path
        scene_params["environment_path"] = environment_path

        self.scene_params = copy.deepcopy(scene_params)
        self.planner_params = planner_params
        del self.scene_params["benchmark_attributes"]
        del self.scene_params["non_benchmark_attributes"]

    def get_assets_path(self, environment_name: str, environment_file_name: str, sdf_file_name: str) -> Tuple[Path, Path]:
        scenes_data_dir_path = Path(self.data_dir_path) / "scenes"
        scene_path = scenes_data_dir_path / environment_name / (environment_file_name + ".urdf")
        sdf_path = scenes_data_dir_path / environment_name / (sdf_file_name + ".sdf")
        assert scene_path.exists(), f"Scene file {scene_path} does not exist"
        assert sdf_path.exists(), f"SDF file {sdf_path} does not exist"
        return scene_path, sdf_path

    def load_parameter_file(self, path: Path):
        """ Load parameter file """
        try:
            with open(path, 'r') as stream:
                params = yaml.safe_load(stream)
        except FileNotFoundError:
            print(f"[Error]: Parameters file {path} could not be found")
            sys.exit('[EXIT]: System will exit, please provide a parameter file and try again')
        except yaml.constructor.ConstructorError as e:
            print(e)
        finally:
            self._params = self.set_params(params)


if __name__ == "__main__":
    parameter_file_path = Path(get_root_package_path()) / "parameters.yaml"
    parameter_loader = ParameterLoader(parameter_file_path)
    parameter_loader.initialize()
    print(parameter_loader.params)
    # print(parameter_loader.params["robot_params"])
    # print(parameter_loader.params["scene_params"])
    # print(parameter_loader.params["planner_params"])
    # print(parameter_loader.params["trainable_params"])
    # print(parameter_loader.params["graphics_params"])
