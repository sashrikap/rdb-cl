import gym
import copy
import numpy as onp
import rdb.envs.drive2d
from time import time, sleep
from tqdm import tqdm, trange
from rdb.optim.mpc import build_mpc
from rdb.infer import *

# ENV_NAME = "Week6_01-v0"  # Blockway
# ENV_NAME = "Week6_01-v1"
# ENV_NAME = "Week6_02-v1"  # Two Blockway
ENV_NAME = "Week6_03-v1"  # Two Blockway
ROOT_DIR = "data/mp4"
N_VIDEOS = 10

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
controller, runner = build_mpc(
    env, main_car.cost_runtime, horizon, env.dt, replan=False, T=T
)

# Run thumbnail
task = env.all_tasks[onp.random.randint(0, num_tasks)]
env.set_task(task)
env.reset()
state = env.state

t_start = time()
actions = controller(state, weights=weights, batch=False)
for i in trange(N_VIDEOS):
    path = f"{ROOT_DIR}/env_{ENV_NAME}/mp4_test_{i:02d}.mp4"
    actions = controller(state, weights=weights, batch=False)
    runner.collect_mp4(env.state, actions, path=path)

print(f"Took {((time() - t_start) / N_VIDEOS):03f}s per video")
