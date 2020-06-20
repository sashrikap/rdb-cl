import gym
import os
import copy
import numpy as onp
import rdb.envs.drive2d
from time import time, sleep
from tqdm import tqdm, trange
from rdb.optim.mpc import build_mpc
from rdb.exps.utils import Profiler, examples_dir, load_params
from rdb.infer import *


ENV_NAME = "Week6_02-v1"  # Two Blockway
FILE_NAME = "Week6_04_1v2"

THUMBNAIL = True
HEATMAP = True
BOUNDMAP = True
CONSTRAINTSMAP = True

env = gym.make(ENV_NAME)
env.reset()
feat_keys = env.features_keys
main_car = env.main_car
horizon = 10
T = 10

num_tasks = len(env.all_tasks)
print(f"Total tasks {num_tasks}")
optimizer, runner = build_mpc(
    env, main_car.cost_runtime, horizon, env.dt, replan=False, T=T
)


tasks = load_params(f"{examples_dir()}/tasks/{FILE_NAME}.yaml")["TASKS"]
for ti, task in enumerate(tasks):
    env.set_task(task)
    env.reset()
    path = f"{examples_dir()}/tasks/{FILE_NAME}/task_{ti:02d}.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    runner.collect_thumbnail(env.state, path=path, close=False, paper=True)
