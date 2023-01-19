import gym
import jax
import copy
import jax.numpy as jnp
import numpy as onp
import rdb.envs.drive2d

from time import time, sleep
from tqdm import tqdm
from rdb.optim.mpc import build_mpc
from rdb.optim.runner import Runner
from rdb.infer import *
from rdb.visualize.render import render_env
from rdb.visualize.preprocess import normalize_features

DRAW_HEAT = False
ENV_NAME = "Week6_02-v1"  # Two Blockway
TASK = (-0.7, -0.7, 0.13, 0.4, -0.13, 0.4)

env = gym.make(ENV_NAME)
main_car = env.main_car
horizon = 10
weights = {
    "dist_cars": 5,
    "dist_lanes": 5,
    "dist_fences": 0.35,
    "dist_objects": 10.25,
    "speed": 5,
    "control": 0.1,
}
env.set_task(TASK)
obs = env.reset()
env.render("human", draw_heat=DRAW_HEAT, weights=weights)

done = False
t = 0

while not done:
    obs, rew, done, truncated, info = env.step(env.action_space.sample())
    t += 1
    # env.render("human", draw_heat=DRAW_HEAT, weights=weights)
    env.render("human")
    sleep(0.2)

import pdb; pdb.set_trace()