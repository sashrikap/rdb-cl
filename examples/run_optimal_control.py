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

DUMMY_ACTION = False
DRAW_HEAT = False
REPLAN = False
MAKE_MP4 = False
ENGINE = "scipy"
METHOD = "lbfgs"
ENV_NAME = "Week6_02-v1"  # Two Blockway
TASK = (-0.7, -0.7, 0.13, 0.4, -0.13, 0.4)

env = gym.make(ENV_NAME)
env.reset()
main_car = env.main_car
horizon = 10
T = 10
weights = {
    "dist_cars": 5,
    "dist_lanes": 5,
    "dist_fences": 0.35,
    "dist_objects": 10.25,
    "speed": 5,
    "control": 0.1,
}
# weights = {"dist_lanes": 5.35}
if TASK == "RANDOM":
    num_tasks = len(env.all_tasks)
    print(f"Total tasks {num_tasks}")
    TASK = env.all_tasks[onp.random.randint(0, num_tasks)]
env.set_task(TASK)
env.reset()

print(f"Task {TASK}")

if not REPLAN:
    T = horizon
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
t1 = time.time()
w_list = DictList([weights])
w_list = w_list.prepare(env.features_keys)
actions = optimizer(state, weights=None, weights_arr=w_list.numpy_array())
traj, cost, info = runner(state, actions, weights=weights, batch=False)
print("cost", cost)
env.reset()
env.render("human", draw_heat=DRAW_HEAT, weights=weights)

for t in range(T):
    env.step(actions[:, t])
    env.render("human", draw_heat=DRAW_HEAT, weights=weights)
    sleep(0.2)


if MAKE_MP4:
    pathname = f"data/run_highway.mp4"
    render_env(env, state, actions, 10, pathname)
