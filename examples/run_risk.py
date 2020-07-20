import gym
import jax
import copy
import jax.numpy as np
import numpy as onp
import rdb.envs.drive2d

from time import time, sleep
from tqdm import tqdm
from rdb.optim.mpc_risk import build_risk_averse_mpc
from rdb.optim.mpc import build_mpc
from rdb.optim.runner import Runner
from rdb.infer import *
from rdb.visualize.render import render_env
from rdb.visualize.preprocess import normalize_features

DUMMY_ACTION = False
DRAW_HEAT = False
REPLAN = 5
ENGINE = "scipy"
METHOD = "lbfgs"
ENV_NAME = "Week6_02-v1"  # Two Blockway
TASK = (-0.7, -0.7, 0.13, 0.4, -0.13, 0.4)

env = gym.make(ENV_NAME)
env.reset()
main_car = env.main_car
horizon = 10
T = 10
weights1 = {
    "dist_cars": 5,
    "dist_lanes": 5,
    "dist_fences": 0.35,
    "dist_objects": 10.25,
    "speed": 5,
    "control": 0.1,
}
weights2 = {
    "dist_cars": 15,
    "dist_lanes": 15,
    "dist_fences": 1.05,
    "dist_objects": 30.75,
    "speed": 15,
    "control": 0.3,
}
env.set_task(TASK)
env.reset()

print(f"Task {TASK}")

optimizer, runner = build_risk_averse_mpc(
    env,
    main_car.cost_runtime,
    horizon,
    env.dt,
    replan=REPLAN,
    T=T,
    engine=ENGINE,
    method=METHOD,
)
state = copy.deepcopy(env.state)
w_list = DictList([weights1, weights2])
actions = optimizer(state, weights=w_list)
actions = optimizer(state, weights=w_list)
print("actions", actions.mean())

optimizer, runner = build_risk_averse_mpc(
    env,
    main_car.cost_runtime,
    horizon,
    env.dt,
    replan=REPLAN,
    T=T,
    engine=ENGINE,
    method=METHOD,
    cost_args={"mode": "trajwise"},
)
state = copy.deepcopy(env.state)
w_list = DictList([weights1, weights2])
actions = optimizer(state, weights=w_list)
actions = optimizer(state, weights=w_list)
print("actions trajwise", actions.mean())

optimizer, runner = build_mpc(
    env,
    main_car.cost_runtime,
    horizon,
    env.dt,
    replan=REPLAN,
    T=T,
    engine=ENGINE,
    method=METHOD,
)
state = copy.deepcopy(env.state)
w_list = DictList([weights2])
actions = optimizer(state, weights=w_list)
print("actions normal", actions.mean())
