import argparse

parser = argparse.ArgumentParser()
parser.add_argument("infile", type=argparse.FileType("r"))
parser.add_argument("outfile", type=argparse.FileType("w"))
args = parser.parse_args()

import itertools as it
import yaml

input_info = yaml.load(args.infile)
problemset_info = {}
problemset_info["scene"] = input_info["env_file"]
problemset_info["robot_name"] = input_info["robot_name"]
ACTIVE_JOINTS = problemset_info["active_joints"] = input_info["active_joints"]
ACTIVE_AFFINE = problemset_info["active_affine"] = input_info["active_affine"]
JOINT_NAMES = problemset_info["joint_names"] = input_info["joint_names"]
STATES = input_info["states"]
DEFAULT_JOINT_VALUES = input_info["default_joint_values"]
INITIAL_POSE = input_info["default_base_pose"]

goals = input_info["problems"]


def save_problemset(problemset, scene):
    Dumper = yaml.SafeDumper
    Dumper.ignore_aliases = lambda self, data: True
    yaml.dump(scene, args.outfile, Dumper=Dumper)
    for queries_id, problems in problemset["queries"].items():
        yaml.dump(queries_id, args.outfile, Dumper=Dumper)
        # for start_state, goal_state in problems.items():
        #     yaml.dump(start_state, args.outfile, Dumper= Dumper)
        #     yaml.dump(".", args.outfile, Dumper= Dumper)
        #     yaml.dump(goal_state, args.outfile, Dumper= Dumper)
        #     yaml.dump(".", args.outfile, Dumper= Dumper)
    # yaml.dump(problemset_info2, args.outfile, Dumper = Dumper)


def gen_problem(goal_states, start_states):
    Dumper = yaml.SafeDumper
    Dumper.ignore_aliases = lambda self, data: True
    for key, value in goal_states.items():
        print(value)
    # yaml.dump(scene, args.outfile, Dumper= Dumper)


def generate_problemset():
    iter = 0
    prob_info = {}
    scene = problemset_info["scene"].split('.')[0]
    prob_info["queries"] = {}
    for (start, goal) in it.combinations(STATES, 2):
        start_states = dict(zip(ACTIVE_JOINTS, *start.values()))
        goal_states = dict(name='joint_constraint')
        goal_states.update(dict(zip(ACTIVE_JOINTS, *goal.values())))
        prob_info["queries"]["pose" + str(iter)] = {}
        prob_info["queries"]["pose" + str(iter)]["start"] = start_states
        prob_info["queries"]["pose" + str(iter)]["goal"] = goal_states
        iter += 1
    Dumper = yaml.SafeDumper
    Dumper.ignore_aliases = lambda self, data: True
    # yaml.dump(STATES, args.outfile, Dumper= Dumper)
    start_config = dict(name='start')
    print(ACTIVE_JOINTS)
    for state in STATES:
        start_config["start." + list(state.keys())[0]] = {}
        start_config["start." + list(state.keys())[0]].update(dict(zip(ACTIVE_JOINTS, *state.values())))
    yaml.dump(start_config, args.outfile, Dumper=Dumper)
    # save_problemset(prob_info, scene)
    # gen_problem(prob_info["queries"], start_states)


if __name__ == "__main__":
    generate_problemset()
