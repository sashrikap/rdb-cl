import gym
import time, copy
import numpy as np
import rdb.envs.drive2d
from tqdm import tqdm
from rdb.optim.mpc import shooting_method
from rdb.optim.runner import Runner
from rdb.visualize.plot import plot_3d, plot_episode
from rdb.visualize.save import save_rewards
from rdb.visualize.preprocess import normalize_features
from rdb.infer.ird_oc import PGM, IRDOptimalControl


## Handles
VIZ_TRAINING = False
FRAME_WIDTH = 450

## Environment setup
env = gym.make("Week3_02-v0")
env.reset()
main_car = env.main_car
horizon = 10
T = 20
optimizer, runner = shooting_method(env, main_car.cost_runtime, horizon, env.dt, T=T)

## Training environments
weights = {
    "dist_cars": 100.0,
    "dist_lanes": 10.0,
    "dist_fences": 300.0,
    "speed": 20.0,
    "control": 80.0,
}
train_pairs = [(0.4, -0.2), (0.7, 0.2)]

for idx, (y0, y1) in enumerate(train_pairs):
    env.set_task(y0, y1)
    env.reset()
    state = env.state

    actions = optimizer(state, weights=weights)
    traj, cost, info = runner(state, actions, weights=weights)
