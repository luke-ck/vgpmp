import sys
import time
from collections import defaultdict
from typing import List, Union, Tuple

from .miscellaneous import suppress_stdout
from .ops import *

__all__ = 'robot'


# <---------------- robot class ----------------->
class Robot:
    """
    Container class for the robot model which is rendered in the simulator.
    The class contains convenience methods for setting and getting the robot state,
    as well as methods for computing the forward kinematics. Joint state configs can be
    get, set, as well as followed. FK here is used mostly for debugging purposes. For the
    implementation used during optimization check the sampler.
    Currently we only support the URDF format for the robot model.
    """

    def __init__(self, params):
        """Initiates constants and a SerialManipulator object based on choosen robot"""

        self.dof = None
        self.sphere_link_interval = None
        self.sphere_link_idx = None
        self.sphere_offsets = None
        self.base_pose = None
        self.joint_to_link = None
        self.joints = None
        self.active_joints = None
        self.joint_limits = None
        self.joint_idx = None
        self.num_spheres = None
        self.active_links = None
        self.neutral_config = None
        self.joint_link_offsets = None
        self.curr_config = None

        self.DH = np.array(params["dh_parameters"]).reshape((-1, 3))
        self.twist = np.array(params["twist"]).reshape((-1, 1))

        self.active_link_idx = []
        self.rs = params['radius']
        self.link_names = params["spheres_over_links"]
        self.urdf_path = params["urdf_path"]
        self.base = params["basename"]
        self.wrist_test = params["wrist_test"]
        self.sphere_link_interval = []  # this is an array of the same size as sphere_link_idx

        start_pos = [0, 0, 0]
        start_orientation = [0.0, 0.0, 0.0, 1.0]
        with suppress_stdout():
            self.robot_model = p.loadURDF(
                self.urdf_path,
                start_pos,
                start_orientation,
                useFixedBase=1
            )

        self.link_idx = self.get_link_idx()
        assert self.link_idx is not None
        assert self.link_idx[self.base] is not None, "Base link not found. Check config file"
        assert self.link_idx[self.wrist_test] is not None, "Wrist link not found. Check config file"
        self.base_idx = self.link_idx[self.base]
        self.wrist_idx = self.link_idx[self.wrist_test]

    def initialise(self,
                   position,
                   orientation,
                   start_config,
                   active_joints,
                   sphere_links,
                   joint_names,
                   default_pose,
                   benchmark: bool
                   ):

        assert active_joints is not None and active_joints != [], "No active joints specified"

        if position and orientation:
            self.reset_pos_and_orn(position, orientation)

        self.set_active_joints(active_joints)

        if benchmark:
            self.set_scene(joint_names, default_pose)
        self.set_active_links(sphere_links)
        self.set_active_link_idx()

        self.set_joint_idx()
        self.set_joint_names()
        self.init_mapping_links_to_spheres()
        self.enable_collision_active_links(0)
        time.sleep(1)  # wait for the robot to settle
        if start_config is not None:
            if isinstance(start_config, list):
                start_config = np.array(start_config)
            self.set_curr_config(start_config)
        self.init_base_pose()
        time.sleep(1)
        self.set_joint_link_frame_offset()

    def enable_collision_active_links(self, mask: int = 0):
        """
        disable collisions for the arm so the simulator doesn't crash when sampling a joint configuration in collision
        params: mask[int]: boolean
        """
        armLinkIndex = []
        # this will disable collision for all links in the arm
        for idx in self.link_idx.keys():
            if "r_" in idx:
                armLinkIndex.append(self.link_idx[idx])

        group = 0
        for idx in armLinkIndex:
            p.setCollisionFilterGroupMask(self.robot_model, idx, group, mask)

    def init_mapping_links_to_spheres(self):
        """
        This function maps the links to the spheres. It is used when computing forward kinematics
        """
        self.sphere_link_idx, total_spheres = self.get_sphere_id()
        assert total_spheres == len(self.rs)
        assert self.num_spheres is None
        # TODO: check if link indexes for spheres coincide with joint indexes

        cumsum = 0
        self.num_spheres = []
        for k, v in self.sphere_link_idx.items():
            # for each link fitted with spheres build an interval denoting which sphere's indexes correspond to that
            # link
            self.sphere_link_interval.append([cumsum, len(v) + cumsum])
            self.num_spheres.append(len(v))
            cumsum += len(v)

    def init_base_pose(self):
        base_pose = self.get_base_pos()
        self.base_pose = base_pose

    def set_active_link_idx(self):
        for link in self.active_links:
            self.active_link_idx.append(self.link_idx[link])

    def set_scene(self, initial_config_names, default_pose):
        assert initial_config_names != [] and initial_config_names is not None
        assert default_pose != [] and default_pose is not None
        for (name, val) in zip(initial_config_names, default_pose):
            idx = self.get_joint_idx_from_name(name)
            assert idx != -1, "For some reason pybullet throws an error when trying to set joint state of the base"
            self.set_joint_config(idx, val)

    def set_active_links(self, sphere_links):
        self.active_links = sphere_links

    def get_base_pos(self) -> np.array:
        base_pos, base_rot = p.getBasePositionAndOrientation(self.robot_model)
        base_rot = quat_to_rotmat(base_rot).reshape((-1, 1))
        return get_base(base_rot, base_pos)

    def get_joints_info(self):
        return [p.getJointInfo(self.robot_model, i) for i in range(0, p.getNumJoints(self.robot_model))]

    def set_curr_config(self, config: np.ndarray):
        assert self.joint_idx is not None
        if config.ndim == 2:
            assert config.shape[0] == 1
            config = np.squeeze(config)
        self.curr_config = config

        self.set_joint_position(self.curr_config)
        p.stepSimulation()

    def set_joint_idx(self):
        self.joint_idx = []
        for i in self.active_joints:
            idx = self.get_joint_idx_from_name(i)
            if idx == -1:
                print(f"Could not find the corresponding index for joint name {i}. There is a mismatch between naming "
                      f"given in the parameter file and actual robot joints. This can give rise to unexpected "
                      f"behaviour, so please edit the parameter file such that this message doesn't appear.")
                sys.exit(-1)
            self.joint_idx.append(idx)

        self.dof = len(self.joint_idx)
        self.set_joints_to_links()

    def set_joints_to_links(self):
        # TODO: make map of joint ids to link ids
        assert len(self.active_link_idx) == len(self.joint_idx)

        self.joint_to_link = {}
        for i in range(len(self.active_link_idx)):
            self.joint_to_link[self.joint_idx[i]] = self.active_link_idx[i]

    def set_joint_names(self):
        self.joints = [p.getJointInfo(self.robot_model, joint)[
                           1] for joint in range(p.getNumJoints(self.robot_model))]

    def get_joint_names(self):
        return self.joints

    def set_active_joints(self, active_joints):
        self.active_joints = active_joints

    def set_joint_limits(self, joint_limits):
        self.joint_limits = joint_limits

    def get_active_joints(self):
        return self.active_joints

    def get_joint_idx_from_name(self, name: str) -> int:
        """
        For a given joint name, return its index in the kinematic chain of the robot.
        Args:
            name(str): name of the joint to return index of
        Returns:
            joint_idx(int): index of the joint. If not found returns -1
        """
        joint_idx = -1
        for j in range(p.getNumJoints(self.robot_model)):
            joint_info = p.getJointInfo(self.robot_model, j)
            if joint_info[1].decode('utf-8') == name:
                joint_idx = joint_info[0]

        return joint_idx

    def set_joint_config(self, idx, targets):
        p.resetJointState(self.robot_model, idx, targets)

    def get_link_idx(self):
        _link_name_to_index = {p.getBodyInfo(
            self.robot_model)[0].decode('UTF-8'): -1, }

        for _id in range(p.getNumJoints(self.robot_model)):
            _name = p.getJointInfo(self.robot_model, _id)[12].decode('UTF-8')
            _link_name_to_index[_name] = _id

        return _link_name_to_index

    def set_joint_position(self, joint_config: Union[List[float], np.array]):
        r"""Set joint angles to active joints (overrides physics engine)
        Args:
            joint_config (array): joint angles to be set
        """
        if isinstance(joint_config, np.ndarray) and joint_config.ndim == 2:
            assert joint_config.shape[0] == 1
            joint_config = np.squeeze(joint_config)
        ceva = {1, 2, 3, 4, 5, 6, 7}
        for idx, joint in enumerate(self.joint_idx):
            p.resetJointState(self.robot_model, joint, joint_config[idx])
        #FOR WAM
        # for i in range(23):
        #     if i not in ceva:
        #         p.resetJointState(self.robot_model, i, 0)
    def get_curr_config(self) -> np.ndarray:
        r"""
        Return joint position for each joint as a list
        """
        return np.array([joint[0] for joint in p.getJointStates(self.robot_model, self.joint_idx)],
                        dtype=np.float64).reshape((1, len(self.joint_idx)))
        
    def set_joint_motor_control(self, position, kp=300, kv=0.5):

        for i, idx in enumerate(self.joint_idx):
            p.setJointMotorControl2(self.robot_model,
                                    idx,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=position[i],
                                    force=kp,
                                    maxVelocity=kv)

    def move_to_next_config(self, next_pos):
        r"""
            Args:
                next_pos(np.array):  (len(self.joint_idx), 1)
            Return:
                success(bool): Whether next joint config in trajectory has been reached or not
        """
        cur_pos = np.array(self.get_curr_config())
        delta = next_pos - cur_pos
        eps = 0.05
        success = 1
        iteration = 0
        while np.max(np.abs(delta)) > eps:

            self.set_joint_motor_control(next_pos, 300, 0.5)
            p.stepSimulation()
            cur_pos = np.array(self.get_curr_config())

            delta = next_pos - cur_pos
            # if iteration % 100 == 0:
            # print(cur_pos)
            # print(f"delta at iter {iteration} is {np.max(np.abs(delta))}")
            iteration += 1
            if iteration > 2000:
                success = 0
                self.set_joint_motor_control(np.squeeze(self.get_curr_config()), 0, 0)
                p.stepSimulation()
                break
        return success

    def move_to_ee_config(self, joint_config):
        r"""
            Args:
            joint_config(np.array): (len(joint_config), len(self.joint_idx))
        """
        success = 1

        for idx, next_pos in enumerate(joint_config):
            # print(f"Current goal joint state to be reached has index {idx}")
            success = self.move_to_next_config(next_pos)
            if not success:
                print(f"Next state was not reachable")
                self.set_joint_motor_control(np.squeeze(self.get_curr_config()), 0, 0)
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

    def forward_kinematics(self, thetas, craig) -> np.array:

        T00 = self.base_pose
        angles = thetas + self.twist
        transform_matrices = np.zeros((len(thetas), 4, 4))
        DH_mat = np.concatenate([angles, self.DH], axis=-1)
        for idx, params in enumerate(DH_mat):
            if craig:
                transform_matrix = get_transform_matrix_craig(params[0], params[1], params[2], params[3])
            else:
                transform_matrix = get_transform_matrix(params[0], params[1], params[2], params[3])
            transform_matrices[idx] = transform_matrix

        homogenous_transforms = np.zeros((len(thetas) + 1, 4, 4), dtype=np.float64)
        homogenous_transforms[0] = T00
        for i in range(len(transform_matrices)):
            homogenous_transforms[i + 1] = np.array(
                homogenous_transforms[i] @ transform_matrices[i]).reshape(4, 4)
        return homogenous_transforms

    def compute_joint_positions(self, joint_config, craig) -> np.array:
        pos_aux = self.forward_kinematics(joint_config, craig)
        pos = pos_aux[1:, :3, 3]
        return pos, pos_aux

    def compute_joint_link_frame_offset(self) -> np.array:
        r"""
            Compute the link frame offset for each link in the robot.
            Returns:
                [np.array]: (len(self.link_idx), 3)
        """
        assert self.active_link_idx is not None
        return np.array(
            [
                np.array(_coord[2]).astype(np.float64)
                for _coord in p.getLinkStates(self.robot_model, self.active_link_idx, computeForwardKinematics=True)
            ], dtype=np.object).astype(np.float64)

    def get_sphere_transform(self, joints):
        # Union[np.array, TensorLike]
        r""" compute the sphere translational transform from joint world frame positions in cartesian coordinates
        Args:
            joints (array): joint cartesian coordinates
        Returns:
            [np.array]: sphere cartesian coordinates
        """
        return np.array([list(get_world_transform(joint, sphere)[0]) for i, joint in enumerate(joints) for sphere in
                         self.sphere_link_idx[self.joint_idx[i]].values()])

    def _reset_joints_fk(self, joint_config):
        r"""
            Ugly. Possibly rewrite this. If x is the parameter joint_config, ▼x a
            soft shift for x (in radians). This function calls set_joint_position(x) to
            set desired joint_config to active joints (overrides pybullet physics),
            and set_joint_motor_control(x + ▼x) to apply very small velocities
            on each element of x (since the physics engine was disabled rotations
            and velocities will be 0). Usually after this we query the link
            (links are attached to joints x) states and compute forward kinematics
            implicitly.
            Args:
                joint_config([np.array]):  [len(joint_idx) x 1]
        """
        self.set_joint_position(joint_config)
        # end_target = [0.01 + x for x in joint_config]
        # self.set_joint_motor_control(end_target)
        p.stepSimulation()

    def set_joint_link_frame_offset(self) -> None:
        link_offset = self.compute_joint_link_frame_offset()
        self.joint_link_offsets = link_offset.reshape((-1, 3))

    def reset_pos_and_orn(self, pos, orn):
        # TODO: implement checks on type of input orientation (as rotation matrix or quaternion or euler)
        if len(orn) == 3:
            orn = p.getQuaternionFromEuler(orn)
        p.resetBasePositionAndOrientation(self.robot_model, pos, orn)
