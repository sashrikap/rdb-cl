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
from rdb.infer.dictlist import *
from PIL import Image

DUMMY_ACTION = False
DRAW_HEAT = False
REPLAN = False
MAKE_MP4 = True
ENGINE = "scipy"
METHOD = "lbfgs"
ENV_NAME = "Week8_02"
TASK = (0, 0)

env = gym.make(ENV_NAME)
env.reset()
main_car = env.main_car
horizon = 30
T = 10
weights = {
    "dist_cars": 10.25,
    "dist_lanes": 0.1,
    "dist_fences": 0,
    "dist_obstacles": 10.25,
    "dist_trees": 12,
    "speed": 0.1,
    "control": 0.1,
}

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
imgs[0].save(f"render_optimal_control_{ENV_NAME}.gif", save_all = True, append_images=imgs[1:], duration=100, loop=0)
