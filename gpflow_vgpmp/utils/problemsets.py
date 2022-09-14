import sys

__all__ = ('problemsets')


def active_affine(problemset):
    if problemset == 'countertop':
        return 0
    if problemset == 'bookshelves':
        return 0
    if problemset == 'industrial':
        return 0
    if problemset == 'industrial2':
        return 0
    if problemset == 'tunnel':
        return 0
    else:
        print("Unknown problem set")
        sys.exit()


def active_sphere_links():
    return ['r_shoulder_pan_link', 'r_shoulder_lift_link', 'r_upper_arm_roll_link',
            'r_elbow_flex_link', 'r_forearm_roll_link', 'r_wrist_flex_link', 'r_wrist_roll_link']


def active_joints(problemset):
    if problemset == 'countertop':
        return ['r_shoulder_pan_joint', 'r_shoulder_lift_joint', 'r_upper_arm_roll_joint',
                'r_elbow_flex_joint', 'r_forearm_roll_joint', 'r_wrist_flex_joint', 'r_wrist_roll_joint']
    if problemset == 'bookshelves':
        return ['r_shoulder_pan_joint', 'r_shoulder_lift_joint', 'r_upper_arm_roll_joint',
                'r_elbow_flex_joint', 'r_forearm_roll_joint', 'r_wrist_flex_joint', 'r_wrist_roll_joint']
    if problemset == 'industrial':
        return ['r_shoulder_pan_joint', 'r_shoulder_lift_joint', 'r_upper_arm_roll_joint',
                'r_elbow_flex_joint', 'r_forearm_roll_joint', 'r_wrist_flex_joint', 'r_wrist_roll_joint']
    if problemset == 'industrial2':
        return ['r_shoulder_pan_joint', 'r_shoulder_lift_joint', 'r_upper_arm_roll_joint',
                'r_elbow_flex_joint', 'r_forearm_roll_joint', 'r_wrist_flex_joint', 'r_wrist_roll_joint']
    if problemset == 'tunnel':
        return ['r_shoulder_pan_joint', 'r_shoulder_lift_joint', 'r_upper_arm_roll_joint',
                'r_elbow_flex_joint', 'r_forearm_roll_joint', 'r_wrist_flex_joint', 'r_wrist_roll_joint']
    else:
        print("Unknown problem set")
        sys.exit()


def default_base_pose(problemset):
    if problemset == 'countertop':
        return [0.03074371707199644, -0.0, -0.0, -0.9995273002077517, 1.0746809244155884,
                1.500715732574463, 0.0]
    if problemset == 'bookshelves':
        return [1.0, 0.0, 0.0, 0.0, 0.11382412910461426, 0.0, 0.0]
    if problemset == 'industrial':
        return [0.9999554696772218, 0.0, 0.0, 0.009437089731840358, 0.10682493448257446,
                -0.09225612878799438, 0.0]
    if problemset == 'industrial2':
        return [0.9074254167930225, 0.0, 0.0, -0.42021317561210453, -0.0048143863677978516,
                0.039366304874420166, 0.0]
    if problemset == 'tunnel':
        return [0.9999523740218402, 0.0, 0.0, 0.00975959466810749, -0.013043247163295746,
                -0.2435353398323059, 0.0]
    else:
        print("Unknown problem set")
        sys.exit()


def default_joint_values(problemset):
    if problemset == 'countertop':
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.16825, 0.0, 0.0, 0.0, -0.8451866735633127, 0.08128204585620136, -1.8311522093787793,
                -1.5550420276338315, 2.6911049983477024, -
                0.8325789594278508, 2.067980015738538,
                2.248201624865942e-15, 0.0, 0.0, 0.0, 0.0, 0.0, -
                1.1102230246251565e-16, 1.5301977050596074,
                -0.1606585554274158, 1.122, -
                2.1212501013292435, 2.5303972717977263, -0.7028113314291433,
                1.925250846169634, -7.494005416219807e-15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if problemset == 'bookshelves':
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.16825, 0.0, 0.0, 0.0, -0.5652894131595758, -
                0.1940789551546196, -1.260201738335192,
                -0.7895653603354864, -2.322747882942366, -
                0.3918504494615993, -2.5173485998351066,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.1347678577910827, 0.05595277251194286, 0.48032314980402596,
                -2.0802263633096487, 1.2294916701952125, -
                0.8773017824611689, 2.932954218704465,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if problemset == 'industrial':
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.16825, 0.0, 0.0, 0.0, 0.4521781848795534, -
                0.06875607890205337, 0.45157478971034287,
                -1.356432823197765, 2.63664188822003, -
                1.2884530258626397, 2.7514905004419816, 2.248201624865942e-15,
                0.0, 0.0, 0.0, 0.0, 0.0, -
                1.1102230246251565e-16, 1.1473791555597268, -0.2578419004077155,
                0.5298918609954418, -2.121201719392923, 2.198118788614387, -1.4189668927954484,
                2.1828521334438378, -4.08006961549745e-15, 0.0, 0.0, 0.0, 0.0, 0.0, 7.549516567451064e-15,
                0.0]
    if problemset == 'industrial2':
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.16825, 0.0, 0.0, 0.0, -0.2761549481623158, 0.8939542514393661, -0.4462472669817504,
                -1.6105027440487039, 2.328001213611822, -
                1.0881333252440992, -2.3998853474978716,
                2.248201624865942e-15, 0.0, 0.0, 0.0, 0.0, 0.0, -
                1.1102230246251565e-16, 1.1473791555597268,
                -0.2578419004077155, 0.5298918609954418, -2.121201719392923, 2.198118788614387,
                -1.4189668927954484, 2.1828521334438378, -1.174060848541103e-14, 0.0, 0.0, 0.0,
                0.0, 0.0, -1.6653345369377348e-16, 0.0]
    if problemset == 'tunnel':
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.16825, 0.0, 0.0, 0.0, -0.37279882212870064, 1.0259015008778194, -0.9875997438281771,
                -1.208229229103619, 2.0676739431952065, -
                0.5630237954661839, -1.6563473012595384,
                2.248201624865942e-15, 0.0, 0.0, 0.0, 0.0, 0.0, -
                1.1102230246251565e-16, 1.1473791555597268,
                -0.2578419004077155, 0.5298918609954418, -2.121201719392923, 2.198118788614387,
                -1.4189668927954484, 2.1828521334438378, -
                1.2961853812498703e-14, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0]
    else:
        print("Unknown problem set")
        sys.exit()


def env_file(problemset):
    if problemset == 'countertop':
        return "data/envs/countertop.env.xml"
    if problemset == 'bookshelves':
        return "data/envs/bookshelves.env.xml"
    if problemset == 'industrial':
        return "data/envs/industrial.env.xml"
    if problemset == 'industrial2':
        return "data/envs/industrial.env.xml"
    if problemset == 'tunnel':
        return "data/envs/tunnel.env.xml"
    if problemset == 'lab':
        return "data/envs/lab.env.xml"
    else:
        print("Unknown problem set")
        sys.exit()


def joint_names(problemset):
    if problemset == 'countertop':
        return ['fl_caster_rotation_joint', 'fl_caster_l_wheel_joint', 'fl_caster_r_wheel_joint',
                'fr_caster_rotation_joint', 'fr_caster_l_wheel_joint', 'fr_caster_r_wheel_joint',
                'bl_caster_rotation_joint',
                'bl_caster_l_wheel_joint', 'bl_caster_r_wheel_joint', 'br_caster_rotation_joint',
                'br_caster_l_wheel_joint',
                'br_caster_r_wheel_joint', 'torso_lift_joint', 'head_pan_joint', 'head_tilt_joint',
                'laser_tilt_mount_joint',
                'r_shoulder_pan_joint', 'r_shoulder_lift_joint', 'r_upper_arm_roll_joint', 'r_elbow_flex_joint',
                'r_forearm_roll_joint', 'r_wrist_flex_joint', 'r_wrist_roll_joint', 'r_gripper_motor_slider_joint',
                'r_gripper_motor_screw_joint', 'r_gripper_l_finger_joint', 'r_gripper_l_finger_tip_joint',
                'r_gripper_r_finger_joint', 'r_gripper_r_finger_tip_joint', 'r_gripper_joint', 'l_shoulder_pan_joint',
                'l_shoulder_lift_joint', 'l_upper_arm_roll_joint', 'l_elbow_flex_joint', 'l_forearm_roll_joint',
                'l_wrist_flex_joint', 'l_wrist_roll_joint', 'l_gripper_motor_slider_joint',
                'l_gripper_motor_screw_joint',
                'l_gripper_l_finger_joint', 'l_gripper_l_finger_tip_joint', 'l_gripper_r_finger_joint',
                'l_gripper_r_finger_tip_joint', 'l_gripper_joint', 'torso_lift_motor_screw_joint']
    if problemset == 'bookshelves':
        return ['torso_lift_motor_screw_joint', 'fl_caster_rotation_joint', 'fl_caster_l_wheel_joint',
                'fl_caster_r_wheel_joint',
                'fr_caster_rotation_joint', 'fr_caster_l_wheel_joint', 'fr_caster_r_wheel_joint',
                'bl_caster_rotation_joint',
                'bl_caster_l_wheel_joint', 'bl_caster_r_wheel_joint', 'br_caster_rotation_joint',
                'br_caster_l_wheel_joint',
                'br_caster_r_wheel_joint', 'torso_lift_joint', 'head_pan_joint', 'head_tilt_joint',
                'laser_tilt_mount_joint',
                'r_shoulder_pan_joint', 'r_shoulder_lift_joint', 'r_upper_arm_roll_joint', 'r_elbow_flex_joint',
                'r_forearm_roll_joint', 'r_wrist_flex_joint', 'r_wrist_roll_joint', 'r_gripper_motor_slider_joint',
                'r_gripper_motor_screw_joint', 'r_gripper_l_finger_joint', 'r_gripper_l_finger_tip_joint',
                'r_gripper_r_finger_joint', 'r_gripper_r_finger_tip_joint', 'r_gripper_joint', 'l_shoulder_pan_joint',
                'l_shoulder_lift_joint', 'l_upper_arm_roll_joint', 'l_elbow_flex_joint', 'l_forearm_roll_joint',
                'l_wrist_flex_joint', 'l_wrist_roll_joint', 'l_gripper_motor_slider_joint',
                'l_gripper_motor_screw_joint',
                'l_gripper_l_finger_joint', 'l_gripper_l_finger_tip_joint', 'l_gripper_r_finger_joint',
                'l_gripper_r_finger_tip_joint', 'l_gripper_joint']
    if problemset == 'industrial':
        return ['fl_caster_rotation_joint', 'fl_caster_l_wheel_joint', 'fl_caster_r_wheel_joint',
                'fr_caster_rotation_joint', 'fr_caster_l_wheel_joint', 'fr_caster_r_wheel_joint',
                'bl_caster_rotation_joint',
                'bl_caster_l_wheel_joint', 'bl_caster_r_wheel_joint', 'br_caster_rotation_joint',
                'br_caster_l_wheel_joint',
                'br_caster_r_wheel_joint', 'torso_lift_joint', 'head_pan_joint', 'head_tilt_joint',
                'laser_tilt_mount_joint',
                'r_shoulder_pan_joint', 'r_shoulder_lift_joint', 'r_upper_arm_roll_joint', 'r_elbow_flex_joint',
                'r_forearm_roll_joint', 'r_wrist_flex_joint', 'r_wrist_roll_joint', 'r_gripper_motor_slider_joint',
                'r_gripper_motor_screw_joint', 'r_gripper_l_finger_joint', 'r_gripper_l_finger_tip_joint',
                'r_gripper_r_finger_joint', 'r_gripper_r_finger_tip_joint', 'r_gripper_joint', 'l_shoulder_pan_joint',
                'l_shoulder_lift_joint', 'l_upper_arm_roll_joint', 'l_elbow_flex_joint', 'l_forearm_roll_joint',
                'l_wrist_flex_joint', 'l_wrist_roll_joint', 'l_gripper_motor_slider_joint',
                'l_gripper_motor_screw_joint',
                'l_gripper_l_finger_joint', 'l_gripper_l_finger_tip_joint', 'l_gripper_r_finger_joint',
                'l_gripper_r_finger_tip_joint', 'l_gripper_joint', 'torso_lift_motor_screw_joint']
    if problemset == 'industrial2':
        return ['fl_caster_rotation_joint', 'fl_caster_l_wheel_joint', 'fl_caster_r_wheel_joint',
                'fr_caster_rotation_joint', 'fr_caster_l_wheel_joint', 'fr_caster_r_wheel_joint',
                'bl_caster_rotation_joint',
                'bl_caster_l_wheel_joint', 'bl_caster_r_wheel_joint', 'br_caster_rotation_joint',
                'br_caster_l_wheel_joint',
                'br_caster_r_wheel_joint', 'torso_lift_joint', 'head_pan_joint', 'head_tilt_joint',
                'laser_tilt_mount_joint',
                'r_shoulder_pan_joint', 'r_shoulder_lift_joint', 'r_upper_arm_roll_joint', 'r_elbow_flex_joint',
                'r_forearm_roll_joint', 'r_wrist_flex_joint', 'r_wrist_roll_joint', 'r_gripper_motor_slider_joint',
                'r_gripper_motor_screw_joint', 'r_gripper_l_finger_joint', 'r_gripper_l_finger_tip_joint',
                'r_gripper_r_finger_joint', 'r_gripper_r_finger_tip_joint', 'r_gripper_joint', 'l_shoulder_pan_joint',
                'l_shoulder_lift_joint', 'l_upper_arm_roll_joint', 'l_elbow_flex_joint', 'l_forearm_roll_joint',
                'l_wrist_flex_joint', 'l_wrist_roll_joint', 'l_gripper_motor_slider_joint',
                'l_gripper_motor_screw_joint',
                'l_gripper_l_finger_joint', 'l_gripper_l_finger_tip_joint', 'l_gripper_r_finger_joint',
                'l_gripper_r_finger_tip_joint', 'l_gripper_joint', 'torso_lift_motor_screw_joint']
    if problemset == 'tunnel':
        return ['fl_caster_rotation_joint', 'fl_caster_l_wheel_joint', 'fl_caster_r_wheel_joint',
                'fr_caster_rotation_joint', 'fr_caster_l_wheel_joint', 'fr_caster_r_wheel_joint',
                'bl_caster_rotation_joint',
                'bl_caster_l_wheel_joint', 'bl_caster_r_wheel_joint', 'br_caster_rotation_joint',
                'br_caster_l_wheel_joint',
                'br_caster_r_wheel_joint', 'torso_lift_joint', 'head_pan_joint', 'head_tilt_joint',
                'laser_tilt_mount_joint',
                'r_shoulder_pan_joint', 'r_shoulder_lift_joint', 'r_upper_arm_roll_joint', 'r_elbow_flex_joint',
                'r_forearm_roll_joint', 'r_wrist_flex_joint', 'r_wrist_roll_joint', 'r_gripper_motor_slider_joint',
                'r_gripper_motor_screw_joint', 'r_gripper_l_finger_joint', 'r_gripper_l_finger_tip_joint',
                'r_gripper_r_finger_joint', 'r_gripper_r_finger_tip_joint', 'r_gripper_joint', 'l_shoulder_pan_joint',
                'l_shoulder_lift_joint', 'l_upper_arm_roll_joint', 'l_elbow_flex_joint', 'l_forearm_roll_joint',
                'l_wrist_flex_joint', 'l_wrist_roll_joint', 'l_gripper_motor_slider_joint',
                'l_gripper_motor_screw_joint',
                'l_gripper_l_finger_joint', 'l_gripper_l_finger_tip_joint', 'l_gripper_r_finger_joint',
                'l_gripper_r_finger_tip_joint', 'l_gripper_joint', 'torso_lift_motor_screw_joint']
    else:
        print("Unknown problem set")
        sys.exit()


def states(problemset):
    if problemset == 'countertop':
        n_states = 9
        states = [0 for i in range(n_states)]
        states[0] = [-0.8452, 0.0813, -1.8312, -
        1.555, 2.6911, -0.8326, 2.068]
        states[1] = [-0.4134, -0.238, -3.6504, -
        1.1768, 2.7225, -1.2706, -2.3709]
        states[2] = [-0.9547, 0.3356, -2.3151, -
        0.9126, 1.8166, -0.8724, -3.1287]
        states[3] = [-0.496, -0.2946, -2.626, -
        1.5671, -0.9644, -0.5307, 0.5828]
        states[4] = [-0.978, -0.235, -1.3629, -
        1.282, -2.2903, -0.4913, 0.9081]
        states[5] = [-0.3043, -0.1995, 0.4997, -
        0.9161, -3.0128, -1.2772, -0.4844]
        states[6] = [-0.0826, -0.3115, -0.7685, -
        1.0468, -2.8332, -1.2915, 0.7087]
        states[7] = [-0.9493, -0.2259, -1.2924, -
        1.2902, -2.2911, -0.5655, -2.1449]
        states[8] = [-0.0077, -0.1813, -1.2825, -
        0.2072, -2.475, -0.3674, -2.5659]
        return n_states, states
    if problemset == 'bookshelves':
        n_states = 10
        states = [0 for i in range(n_states)]
        states[0] = [-0.5653, -0.1941, -1.2602, -
        0.7896, -2.3227, -0.3919, -2.5173]
        states[1] = [-0.1361, -0.1915, -1.2602, -
        0.8652, -2.8852, -0.7962, -2.039]
        states[2] = [0.2341, -0.2138, -1.2602, -
        0.4709, -3.0149, -0.7505, -2.0164]
        states[3] = [0.1584, 0.3429, -1.2382, -
        0.9829, -2.0892, -1.6126, -0.5582]
        states[4] = [0.3927, 0.1763, -1.2382, -
        0.1849, -1.96, -1.4092, -1.0492]
        states[5] = [-0.632, 0.5012, -1.2382, -
        0.8353, 2.2571, -0.1041, 0.3066]
        states[6] = [0.1683, 0.7154, -0.4195, -
        1.0496, 2.4832, -0.6028, -0.6401]
        states[7] = [-0.1198, 0.5299, -0.6291, -
        0.4348, 2.1715, -1.6403, 1.8299]
        states[8] = [0.2743, 0.4088, -0.5291, -
        0.4304, 2.119, -1.9994, 1.7162]
        states[9] = [0.2743, 0.4088, -0.5291, -
        0.4304, -0.9985, -1.0032, -1.7278]
        return n_states, states
    if problemset == 'industrial':
        n_states = 12
        states = [0 for i in range(n_states)]
        states[0] = [0.4522, -0.0688, 0.4516, -
        1.3564, 2.6366, -1.2885, 2.7515]
        states[1] = [0.5645, -0.2949, -0.8384, -
        0.4594, 2.9742, -1.1313, -2.3144]
        states[2] = [0.2846, -0.3534, -2.2822, -
        1.2794, -2.6088, -1.298, -1.1206]
        states[3] = [0.281, 0.7666, 0.0846, -
        1.8104, 2.8603, -1.1027, -3.1227]
        states[4] = [0.0514, -0.3534, -2.8542, -
        1.7384, -2.95, -1.3661, -0.3652]
        states[5] = [0.2874, -0.0535, -2.3339, -
        1.2294, -2.8416, -1.3918, -0.9476]
        states[6] = [0.5259, 0.9754, -0.2409, -
        0.9607, 1.7601, -0.748, -1.6548]
        states[7] = [-0.4355, 1.2821, -0.5184, -
        1.4671, 2.7845, -0.2334, -2.709]
        states[8] = [-1.0536, -0.3436, -1.6586, -
        1.4779, -2.7631, -0.4247, -1.6377]
        states[9] = [-0.7221, -0.3534, -1.639, -
        1.7309, -2.8092, -0.9864, -1.5293]
        states[10] = [-0.3357, -0.1715, -0.3289, -
        2.0001, 1.6523, -1.9265, 2.5474]
        states[11] = [-0.3888, 1.0477, -1.1405, -
        0.7096, 0.9253, -0.5049, -0.3575]
        return n_states, states
    if problemset == 'industrial2':
        n_states = 10
        states = [0 for i in range(n_states)]
        states[0] = [-0.2762, 0.894, -0.4462, -
        1.6105, 2.328, -1.0881, -2.3999]
        states[1] = [-0.571, -0.3534, -1.7684, -
        1.5384, -2.7693, -1.5537, -1.4818]
        states[2] = [-0.1708, 0.093, -1.1193, -
        1.0388, 2.8681, -1.4571, -1.9924]
        states[3] = [0.2855, -0.198, -3.2144, -
        1.0791, -2.0565, -1.1219, -0.6414]
        states[4] = [-0.5471, 0.1817, -1.835, -
        1.6187, 3.0026, -1.7675, -1.3979]
        states[5] = [0.1713, 0.377, -0.5826, -
        0.5771, -1.5264, -0.4143, 2.0764]
        states[6] = [0.514, 0.2662, -1.2524, -
        0.6177, 2.9156, -0.2591, -1.7356]
        states[7] = [0.5512, -0.3535, -1.2124, -
        0.4724, -2.1021, -0.5965, -2.8023]
        states[8] = [0.5272, 0.7193, -0.9876, -
        0.5453, 1.2938, -0.3151, -0.5195]
        states[9] = [-0.3728, 1.0259, -0.9876, -
        1.2082, 2.0042, -1.3781, -1.6173]
        return n_states, states
    if problemset == 'tunnel':
        n_states = 4
        states = [0 for i in range(n_states)]
        states[0] = [-0.3728, 1.0259, -0.9876, -
        1.2082, 2.0677, -0.563, -1.6563]
        states[1] = [0.2993, 0.1747, -1.7835, -
        0.8593, 3.1042, -1.2145, -1.3626]
        states[2] = [0.3537, 0.2079, -1.7168, -
        0.2937, 3.0224, -0.7083, -1.3286]
        states[3] = [0.3398, -0.2349, -0.0415, -
        1.5042, 2.7647, -1.7995, 3.0539]
        return n_states, states
    if problemset == 'lab':
        starts = 4
        ends = 6
        start = [[0 for i in range(7)] for x in range(starts)]
        end = [[0 for i in range(7)] for x in range(ends)]
        start[0] = [1.580, 1.100, 0.000, 1.700, 0.000, -1.240, 1.570]
        start[1] = [-0.605, -1.710, 1.640, 1.290, 1.200, -0.106, 2.000]
        start[2] = [1.650, 1.400, 0.000, 0.900, 0.000, -0.800, 1.570]
        start[3] = [2.100, 1.000, 0.000, 1.700, 0.000, -1.000, 1.570]
        end[0] = [0.000, 0.600, 0.000, 1.300, 0.000, -0.250, 1.570]
        end[1] = [0.260, 0.600, 0.000, 1.300, 0.000, -0.250, 1.570]
        end[2] = [-0.260, 0.600, 0.000, 1.300, 0.000, -0.250, 1.570]
        end[3] = [0.000, 0.970, 0.000, 1.600, 0.000, -0.919, 1.570]
        end[4] = [0.260, 0.970, 0.000, 1.600, 0.000, -0.919, 1.570]
        end[5] = [-0.260, 0.970, 0.000, 1.600, 0.000, -0.919, 1.570]
        return starts, ends, start, end
    else:
        print("Unknown problem set")
        sys.exit()
