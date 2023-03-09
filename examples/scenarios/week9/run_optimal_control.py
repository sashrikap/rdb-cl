import gym
import jax
import copy
import jax.numpy as jnp
import numpy as onp
import rdb.envs.drive2d
import json

from time import time, sleep
from tqdm import tqdm
from rdb.optim.mpc import build_mpc
from rdb.optim.runner import Runner
from rdb.infer import *
from rdb.visualize.render import render_env
from rdb.visualize.preprocess import normalize_features
from rdb.infer.dictlist import *
from PIL import Image

DUMMY_ACTION = False
DRAW_HEAT = False
REPLAN = True
MAKE_MP4 = True
ENGINE = "scipy"
METHOD = "lbfgs"
ENV_NAME = "Week9_03"
TASK = (0, 0, .5, .5)

env = gym.make(ENV_NAME)
env.reset()
main_car = env.main_car
horizon = 10
T = 30
fname = f"{ENV_NAME}_05"
with open(f"../weights/{fname.lower()}.json") as json_file:
    weights = json.load(json_file)

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
t1 = time()
w_list = DictList([weights])
w_list = w_list.prepare(env.features_keys)
actions = optimizer(state, weights=None, weights_arr=w_list.numpy_array())
traj, cost, info = runner(state, actions, weights=weights, batch=False)
print("cost", cost)
env.reset()
env.render("human", draw_heat=DRAW_HEAT, weights=weights)
frames = []

for t in range(T):
    env.step(actions[:, t])
    img = env.render("rgb_array")
    frames.append(img)
    sleep(0.2)


imgs = [Image.fromarray(img) for img in frames]
imgs[0].save(f"render_optimal_control_{fname}.gif", save_all = True, append_images=imgs[1:], duration=100, loop=0)
