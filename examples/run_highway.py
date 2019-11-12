import gym
import time, copy
import jax.numpy as np
import rdb.envs.drive2d

from rdb.optim.open import shooting_optimizer
from rdb.optim.runner import Runner
from rdb.visualize.render import render_env
from rdb.visualize.preprocess import normalize_features

REPLAN = True
MAKE_MP4 = False

env = gym.make("Week3_02-v0")
obs = env.reset()
main_car = env.main_car
udim = 2
horizon = 10

T = 30
if not REPLAN:
    T = horizon
optimizer = shooting_optimizer(
    env.dynamics_fn, main_car.cost_runtime, udim, horizon, env.dt, replan=REPLAN, T=T
)
runner = Runner(env, main_car)

y0_idx, y1_idx = 1, 5
state = copy.deepcopy(env.state)
# state[y0_idx] = 0.4
# state[y1_idx] = -0.2
state[y0_idx] = -0.5
state[y1_idx] = 0.2
# state[y0_idx] = 0.5
# state[y1_idx] = 0.0
env.state = state

weights = {
    "dist_cars": 100.0,
    "dist_lanes": 10.0,
    "dist_fences": 300.0,
    "speed": 20.0,
    "control": 80.0,
}

actions = optimizer(env.state, weights=weights)
traj, cost, info = runner(env.state, actions)

env.render("human")
print(f"Total cost {cost}")

for t in range(T):
    env.step(actions[t])
    env.render("human")
    time.sleep(0.2)

if MAKE_MP4:
    pathname = f"data/y0({state[y0_idx]:.2f})_y1({state[y1_idx]:.2f}) theta 2.mp4"
    render_env(env, state, actions, 10, pathname)
