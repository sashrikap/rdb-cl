import gym
import copy
import numpy as onp
import rdb.envs.drive2d
from time import time, sleep
from tqdm import tqdm, trange
from rdb.optim.mpc import build_mpc
from rdb.infer import *

NUM_THUMBNAILS = 200
# ENV_NAME = "Week6_01-v0"  # Blockway
# ENV_NAME = "Week6_01-v1"
# ENV_NAME = "Week6_02-v1"  # Two Blockway
ENV_NAME = "Week6_03-v1"  # Two Blockway
ROOT_DIR = "data/thumbnails"


THUMBNAIL = True
HEATMAP = False
BOUNDMAP = False
CONSTRAINTSMAP = False

env = gym.make(ENV_NAME)
env.reset()
feat_keys = env.features_keys
main_car = env.main_car
horizon = 10
T = 10

weights = {
    "dist_cars": 1.1,
    "dist_lanes": 0.1,
    "dist_fences": 0.35,
    "dist_objects": 100.25,
    "speed": 0.05,
    "control": 0.1,
}
num_tasks = len(env.all_tasks)
print(f"Total tasks {num_tasks}")
optimizer, runner = build_mpc(
    env, main_car.cost_runtime, horizon, env.dt, replan=False, T=T
)

# Run thumbnail
if THUMBNAIL:
    for ni in trange(NUM_THUMBNAILS):
        task = env.all_tasks[onp.random.randint(0, num_tasks)]
        env.set_task(task)
        env.reset()
        path = f"{ROOT_DIR}/env_{ENV_NAME}/thumbnail_{ni:03d}.png"
        runner.collect_thumbnail(env.state, path=path)


# Run costs
if HEATMAP:
    task = env.all_tasks[onp.random.randint(0, num_tasks)]
    for key in ["dist_cars", "dist_lanes", "dist_fences", "dist_objects"]:
        weight_k = copy.deepcopy(weights)
        for keyi in weight_k.keys():
            if keyi == key:
                weight_k[keyi] = 10
            else:
                weight_k[keyi] = 0
        weight_k = DictList([weight_k])
        weight_k = weight_k.prepare(feat_keys)
        env.set_task(task)
        env.reset()
        path = f"{ROOT_DIR}/env_{ENV_NAME}/heatmap_{key}.png"
        runner.collect_cost_heatmap(env.state, weight_k, path=path)


if BOUNDMAP:
    task = env.all_tasks[onp.random.randint(0, num_tasks)]
    for key in ["dist_cars", "dist_lanes", "dist_fences", "dist_objects"]:
        weight_k = copy.deepcopy(weights)
        for keyi in weight_k.keys():
            if keyi == key:
                weight_k[keyi] = 10
            else:
                weight_k[keyi] = 0
        weight_k = DictList([weight_k])
        weight_k = weight_k.prepare(feat_keys)
        env.set_task(task)
        env.reset()
        path = f"{ROOT_DIR}/env_{ENV_NAME}/boundmap_{key}.png"
        runner.collect_bound_heatmap(env.state, weight_k, path=path)

if CONSTRAINTSMAP:
    task = env.all_tasks[onp.random.randint(0, num_tasks)]
    for key in ["offtrack", "wronglane", "collision", "crash_objects"]:
        env.set_task(task)
        env.reset()
        path = f"{ROOT_DIR}/env_{ENV_NAME}/constraintsmap_{key}.png"
        runner.collect_constraint_heatmap(env.state, key, path=path)
