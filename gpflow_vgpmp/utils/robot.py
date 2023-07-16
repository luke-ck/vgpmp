from collections import defaultdict
from typing import Union, Tuple

from .miscellaneous import suppress_stdout, are_all_elements_integers
from .ops import *
from .parameter_loader import ParameterLoader
from .robot_mixin import RobotMixin

__all__ = 'robot'

from .simulation import Simulation


# <---------------- robot class ----------------->


class Robot(RobotMixin):
    """
    Class for the robot model which is rendered in the simulator (interfaces with pybullet).
    As such, it contains convenience methods for setting and getting the robot state,
    as well as methods for computing the forward kinematics, and making the robot follow a
    certain trajectory. FK here is used mostly for debugging purposes. For the
    implementation used during optimization check the sampler.
    Currently, we only support the URDF format for the robot model.
    """

    def __init__(self, config: ParameterLoader, simulation: Simulation):
        params = config.robot_params
        client = simulation.simulation_thread.client

        super().__init__(**params)
        # TODO: check the case where pybullet silently fails to load the urdf
        self._orientation = None
        self._position = None
        self.num_spheres_per_link = None
        urdf_path = params["urdf_path"]
        assert client == 0, "Only one client is supported at the moment. Undefined behavior for multiple clients."
        assert urdf_path.exists() and urdf_path.is_file(), "URDF file not found"
        # with suppress_stdout(): # this breaks tests. suppress annoying warnings from pybullet.
        self.robot_model = p.loadURDF(
            urdf_path.as_posix(),
            useFixedBase=1,
            # flags=p.URDF_USE_SELF_COLLISION # for some reason it makes wam's fingers to constantly move
            physicsClientId=client
        )
        self._joint_names = None
        self._link_names = None
        self.active_link_names = None
        self.active_link_indexes = None
        self.active_joint_names = None
        self.active_joint_indexes = None
        self.joint_to_link_index_dict = None
        self.joint_index_to_name_dict = None
        self.joint_name_to_index_dict = None
        self.link_name_to_index_dict = None
        self.link_index_to_name_dict = None

        self.num_frames_for_spheres = params["num_frames_for_spheres"]
        self.link_name_base = params["link_name_base"]
        self.link_name_wrist = params["link_name_wrist"]
        self.base_pose = None
        self.joint_link_offsets = None
        self.curr_joint_config = None

        self.sphere_link_interval = []  # this is an array of the same size as sphere_link_idx
        self.sphere_link_idx = None
        self.sphere_offsets = None
        self.sphere_radii = params['radius']
        self.num_spheres = len(self.sphere_radii)

        self.set_joint_names()
        self.set_link_names()

        self.set_link_name_to_index()
        self.set_link_index_to_name()
        self.set_joint_name_to_index()
        self.set_joint_index_to_name()

        self.set_active_joint_names(params["active_joints"])
        self.set_active_link_names(params["active_links"])
        assert self.active_joint_names is not None and self.active_joint_names != [], "No active joints specified"
        assert self.joint_name_to_index_dict is not None
        assert self.joint_name_to_index_dict.keys().isdisjoint(self.active_joint_names) is False, \
            "Active joint names not found in joint name to index dict. Check config file"

        assert self.link_name_to_index_dict is not None
        assert self.link_name_to_index_dict[self.link_name_base] is not None, "Base link not found. Check config file"
        assert self.link_name_to_index_dict[self.link_name_wrist] is not None, "Wrist link not found. Check config file"
        self.base_index = self.link_name_to_index_dict[self.link_name_base]
        self.wrist_index = self.link_name_to_index_dict[self.link_name_wrist]
        assert self.joint_limits is not None, \
            "Must pass joint constraints to initialize the planner. Even if there are no joint constraints, pass " \
            "values such as -3.15, +3.15 for limits. This is because if q_mu is None, the model will initialize " \
            "q_mu as the mean value of the sigmoid bounds."
        self.is_initialized = False

    def initialise(self,
                   default_robot_pos_and_orn,
                   joint_names,
                   default_pose,
                   benchmark: bool
                   ):

        try:
            position, orientation = default_robot_pos_and_orn
            assert len(position) == 3
            assert len(orientation) == 4 or len(orientation) == 3
        except TypeError:
            position, orientation = None, None
        if position and orientation:
            print("Setting robot position and orientation")
            self.reset_pos_and_orn(position, orientation)
        if benchmark:
            self.set_scene(joint_names, default_pose)
        self.init_mapping_links_to_spheres()
        self.enable_collision_active_links(0)
        time.sleep(1)  # wait for the robot to settle
        # if start_config is not None:
        #     if isinstance(start_config, list):
        #         start_config = np.array(start_config)
        #     self.set_curr_joint_config(start_config)
        self.init_base_pose()
        time.sleep(1)
        self.set_joint_link_frame_offset()
        self.is_initialized = True

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, value):
        assert len(value) == 4
        self._orientation = value

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        assert len(value) == 3
        self._position = value

    def enable_collision_active_links(self, mask: int = 0):
        """
        disable collisions for the arm so the simulator doesn't crash when sampling a joint configuration in collision
        params: mask[int]: boolean
        """
        arm_link_index = []
        # this will disable collision for all links in the arm
        for idx in self.link_name_to_index_dict.keys():
            arm_link_index.append(self.link_name_to_index_dict[idx])

        group = 0
        for idx in arm_link_index:
            p.setCollisionFilterGroupMask(self.robot_model, idx, group, mask)

    def init_base_pose(self):
        _base_pose = self.get_base_pose()
        self.base_pose = _base_pose

    def get_base_pose(self) -> np.array:
        base_pos, base_rot = p.getBasePositionAndOrientation(self.robot_model)
        base_rot = quat_to_rotmat(base_rot).reshape((-1, 1))
        return get_base(base_rot, base_pos)

    def reset_pos_and_orn(self, pos, orn):
        # TODO: implement checks on type of input orientation (as rotation matrix or quaternion or euler)

        if len(orn) == 3:
            orn = p.getQuaternionFromEuler(orn)
        p.resetBasePositionAndOrientation(self.robot_model, pos, orn)
        print(pos, orn)
        self.position = pos
        self.orientation = orn

    def get_joints_info(self):
        return [p.getJointInfo(self.robot_model, i) for i in range(0, p.getNumJoints(self.robot_model))]

    def get_joint_index_from_name(self, name: str) -> int:
        """
        For a given joint name, return its index in the kinematic chain of the robot.
        Args:
            name(str): name of the joint to return index of
        Returns:
            joint_idx(int): index of the joint. If not found returns -1
        """
        assert self.joint_name_to_index_dict is not None, "Joint name to index dictionary not initialised"
        return self.joint_name_to_index_dict.get(name, -1)

    def get_attr_name_to_index(self, attr_index=1):
        attr_name_to_index_dict = {}
        for _id in range(p.getNumJoints(self.robot_model)):
            _name = self.decode_joint_info_attribute(_id, attr_index)
            attr_name_to_index_dict[_name] = _id
        return attr_name_to_index_dict

    def decode_joint_info_attribute(self, id, attr_index):
        return p.getJointInfo(self.robot_model, id)[attr_index].decode('UTF-8')

    def set_scene(self, initial_config_joint_names, default_pose):
        assert initial_config_joint_names != [] and initial_config_joint_names is not None
        assert default_pose != [] and default_pose is not None
        for (name, val) in zip(initial_config_joint_names, default_pose):
            idx = self.get_joint_index_from_name(name)
            assert idx != -1, "For some reason pybullet throws an error when trying to set joint state of the base"
            self.set_joint_state(idx, val)

    def set_current_joint_config(self, config: np.ndarray):
        assert self.active_joint_indexes is not None
        if config.ndim == 2:
            assert config.shape[1] == 1 and config.shape[0] == self.dof
            config = np.squeeze(config)
        self.curr_joint_config = config

        self.set_gripper_current_joint_config(self.curr_joint_config)
        p.stepSimulation()

    # ---------------------- Getters and Setters ----------------------
    def get_link_name_to_index(self):
        return self.link_name_to_index_dict

    def set_link_name_to_index(self):
        self.link_name_to_index_dict = self.get_attr_name_to_index(12)

    def get_joint_name_to_index(self):
        return self.joint_name_to_index_dict

    def set_joint_name_to_index(self):
        self.joint_name_to_index_dict = self.get_attr_name_to_index(1)

    def get_link_index_to_name(self):
        return self.link_index_to_name_dict

    def set_link_index_to_name(self):
        assert self.link_name_to_index_dict is not None, "Link name to index dictionary not set"
        self.link_index_to_name_dict = {v: k for k, v in self.link_name_to_index_dict.items()}

    def get_joint_index_to_name(self):
        return self.joint_index_to_name_dict

    def set_joint_index_to_name(self):
        assert self.joint_name_to_index_dict is not None, "Joint name to index dictionary not set"
        self.joint_index_to_name_dict = {v: k for k, v in self.joint_name_to_index_dict.items()}

    def set_active_joints_to_links(self):
        assert len(self.active_link_indexes) == self.dof
        assert len(self.active_joint_indexes) == self.dof

        for joint_index, link_index in zip(self.active_joint_indexes, self.active_link_indexes):
            self.joint_to_link_index_dict[joint_index] = link_index

    def set_joint_state(self, idx: int, target: float):
        p.resetJointState(self.robot_model, idx, target)

    def get_joint_names(self):
        return self._joint_names

    def set_joint_names(self):
        assert self._joint_names is None, "Joint names already set"
        joint_names = [self.decode_joint_info_attribute(joint, 1) for joint in range(p.getNumJoints(self.robot_model))]
        self._joint_names = tuple(joint_names)

    def get_link_names(self):
        return self._link_names

    def set_link_names(self):
        assert self._link_names is None, "Link names already set"
        link_names = [self.decode_joint_info_attribute(joint, 12) for joint in range(p.getNumJoints(self.robot_model))]
        self._link_names = tuple(link_names)

    def get_active_joint_names(self):
        return self.active_joint_names

    def set_active_joint_names(self, active_joint_names=None, on_call=True):
        if on_call:
            assert active_joint_names is not None, "Active joint names not provided"
            assert len(active_joint_names) == self.dof, "Number of active joints must be equal to the number of links"
            self.active_joint_names = active_joint_names
            self.dof = len(self.active_joint_names)
            new_active_joint_indexes = list(map(self.joint_name_to_index_dict.get, active_joint_names))
            self.set_active_joint_indexes(new_active_joint_indexes, on_call=False)
        else:
            self.active_joint_names = active_joint_names

    def get_active_link_names(self):
        return self.active_link_names

    def set_active_link_names(self, active_link_names, on_call=True):
        if on_call:
            assert active_link_names is not None, "Active link names not provided"
            self.active_link_names = active_link_names
            self.dof = len(self.active_link_names)
            new_active_link_indexes = list(map(self.link_name_to_index_dict.get, active_link_names))
            self.set_active_link_indexes(new_active_link_indexes, on_call=False)
        else:
            self.active_link_names = active_link_names

    def get_active_joint_indexes(self):
        return self.active_joint_indexes

    def set_active_joint_indexes(self, active_joint_indexes, on_call=True):
        if on_call:
            assert are_all_elements_integers(
                active_joint_indexes) and active_joint_indexes is not None, "Invalid type of joint indexes. "
            assert len(active_joint_indexes) == len(
                set(active_joint_indexes)), "Duplicate joint indexes found in the list. "
            assert -1 not in active_joint_indexes, "Invalid joint index found in the list of active joint indexes. "

            self.dof = len(active_joint_indexes)
            self.active_joint_indexes = tuple(active_joint_indexes)

            new_active_joint_names = list(map(self.joint_index_to_name_dict.get, active_joint_indexes))
            self.set_active_joint_names(new_active_joint_names, on_call=False)
            self.set_active_joints_to_links()
        else:
            self.active_joint_indexes = active_joint_indexes

    def get_active_link_indexes(self):
        return self.active_link_indexes

    def set_active_link_indexes(self, active_link_indexes=None, on_call=True):
        if on_call:
            assert active_link_indexes is not None, "Active link indexes not provided"
            assert len(active_link_indexes) == self.dof, "Number of active links must be equal to the number of joints"
            self.active_link_indexes = active_link_indexes
            new_active_link_names = list(map(self.link_index_to_name_dict.get, active_link_indexes))
            self.set_active_link_names(new_active_link_names, on_call=False)
        else:
            self.active_link_indexes = active_link_indexes

    def set_joint_link_frame_offset(self) -> None:
        link_offset = self.compute_joint_link_frame_offset()
        self.joint_link_offsets = link_offset.reshape((-1, 3))

    def get_current_joint_config(self) -> np.ndarray:
        r"""
        Return joint position for each joint as a list
        """
        return np.array([p.getJointState(self.robot_model, idx)[0] for idx in self.active_joint_indexes],
                        dtype=np.float64).reshape((1, self.dof))

    def set_gripper_current_joint_config(self, joint_config: Union[List[float], np.array]):
        r"""Set joint angles to active joints (overrides physics engine)
        Args:
            joint_config (array): joint angles to be set
        """
        if isinstance(joint_config, np.ndarray) and joint_config.ndim == 2:
            assert joint_config.shape[0] == 1
            joint_config = np.squeeze(joint_config)

        for idx, joint in enumerate(self.active_joint_indexes):
            self.set_joint_state(joint, joint_config[idx])

        if self.name == "wam":
            active_indices = {1, 2, 3, 4, 5, 6, 7}
            for i in range(23):
                if i not in active_indices:
                    self.set_joint_state(i, 0)

        elif self.name == "franka":
            active_indices = {0, 1, 2, 3, 4, 5, 6}
            for i in range(11):
                if i not in active_indices:
                    self.set_joint_state(i, 0)

    def set_joint_motor_control(self, position, kp=300, kv=0.5):

        for i, idx in enumerate(self.active_joint_indexes):
            p.setJointMotorControl2(self.robot_model,
                                    idx,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=position[i],
                                    force=kp,
                                    maxVelocity=kv)

    def move_to_next_config(self, next_pos):
        r"""
        Move the robot from the current joint configuration to the specified next joint configuration.

        Args:
            next_pos (np.array): Target joint configuration to reach (shape: (dof,))

        Returns:
            success (bool): Whether the next joint configuration in the trajectory has been reached or not
        """
        cur_pos = np.array(self.get_current_joint_config())
        delta = next_pos - cur_pos
        eps = 0.05  # Maximum allowable error (5 cm)

        success = True
        iteration = 0

        # Continue moving until the maximum error is below the threshold
        while np.max(np.abs(delta)) > eps:
            # Set joint motor control to move towards the next joint configuration using position control
            self.set_joint_motor_control(next_pos, kp=500, kv=0.5)

            p.stepSimulation()

            cur_pos = np.array(self.get_current_joint_config())
            delta = next_pos - cur_pos

            iteration += 1

            # Check if the iteration count exceeds a threshold
            if iteration > 2000:
                success = False
                # Stop the joint motion if the maximum iteration count is reached
                self.set_joint_motor_control(np.squeeze(self.get_current_joint_config()), kp=0, kv=0)
                p.stepSimulation()
                break

        return success

    def move_to_ee_config(self, joint_config) -> bool:
        r"""
        Move the robot to a sequence of joint configurations in order to reach the desired end-effector configuration.

        Args: joint_config (np.array): Sequence of joint configurations to reach (shape: (num_configs,
        len(self.joint_idx)))

        Returns:
            success (bool): Whether the end-effector configuration has been reached successfully or not
        """
        success = True

        for idx, next_pos in enumerate(joint_config):
            # Print the current goal joint state index
            # print(f"Current goal joint state to be reached has index {idx}")

            # Move to the next joint configuration in the sequence
            success = self.move_to_next_config(next_pos)

            if not success:
                print(f"Next state was not reachable")
                # Stop the joint motion if the desired configuration is not reachable
                self.set_joint_motor_control(np.squeeze(self.get_current_joint_config()), kp=0, kv=0)
                break

        return success

    def get_sphere_id(self) -> Tuple[defaultdict, int]:
        r"""
        Return: dictionary of sphere ids. The keys are link ids and the values are dictionaries with sphere ids as
        keys and relative sphere frame offsets for values.
        """
        sphere_idx = defaultdict(dict)
        counter = 0
        self.sphere_offsets = []
        for visual_shape in p.getVisualShapeData(self.robot_model):
            if visual_shape[2] == 2:
                sphere_idx[visual_shape[1]].update(
                    {counter: np.array(visual_shape[5])})
                self.sphere_offsets.append(visual_shape[5])
                counter += 1

        self.sphere_offsets = np.array(self.sphere_offsets, dtype=np.float64).reshape((-1, 3))

        return sphere_idx, counter

    def get_link_world_pos(self, idx: Union[List[int], np.array]) -> np.array:
        r"""
            Return the cartesian world position for a given array of link indexes in the robot.
            The position is computed through forward kinematics.
            Args:
                idx([array]):  Any 1-D array of link indexes
            Returns:
                [np.array]: (len(idx), )
        """
        return np.array(
            [
                np.array(_coord[4]).astype(np.float64)
                for _coord in p.getLinkStates(self.robot_model, idx, computeForwardKinematics=True)
            ], dtype=np.object).astype(np.float64)

    def compute_joint_positions(self, joint_config) -> np.array:
        pos_aux = self.forward_kinematics(joint_config)
        pos = pos_aux[1:, :3, 3]
        return pos, pos_aux

    def compute_joint_link_frame_offset(self) -> np.array:
        r"""
            Compute the link frame offset to the joint reference frame for each active link in the robot.
            Returns:
                [np.array]: (len(self.link_idx), 3)
        """
        assert self.active_link_indexes is not None
        return np.array(
            [
                np.array(_coord[2]).astype(np.float64)
                for _coord in p.getLinkStates(self.robot_model, self.active_link_indexes, computeForwardKinematics=True)
            ], dtype=np.object).astype(np.float64)

    def init_mapping_links_to_spheres(self):
        """
        This function maps the links to the spheres. It is used when computing forward kinematics
        """
        self.sphere_link_idx, total_spheres = self.get_sphere_id()
        assert total_spheres == len(self.sphere_radii)
        assert self.num_spheres_per_link is None
        # TODO: check if link indexes for spheres coincide with joint indexes

        cumsum = 0
        self.num_spheres_per_link = []
        for k, v in self.sphere_link_idx.items():
            # for each link fitted with spheres build an interval denoting which sphere's indexes correspond to that
            # link
            self.sphere_link_interval.append([cumsum, len(v) + cumsum])
            self.num_spheres_per_link.append(len(v))
            cumsum += len(v)

    def get_sphere_transform(self, joints):
        # Union[np.array, TensorLike]
        r""" compute the sphere translational transform from joint world frame positions in cartesian coordinates
        Args:
            joints (array): joint cartesian coordinates
        Returns:
            [np.array]: sphere cartesian coordinates
        """
        return np.array([list(get_world_transform(joint, sphere)[0]) for i, joint in enumerate(joints) for sphere in
                         self.sphere_link_idx[self.active_joint_indexes[i]].values()])
