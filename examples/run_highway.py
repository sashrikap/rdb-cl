import gym
import time, copy
import jax.numpy as np
import rdb.envs.drive2d
from rdb.optim.open import shooting_optimizer
from rdb.optim.runner import Runner
from matplotlib import pyplot as plt
from rdb.visualize.render import render_env

REPLAN = True
MAKE_MP4 = False

env = gym.make("Week3_01-v0")
obs = env.reset()
main_car = env.main_car
udim = 2
horizon = 10

T = 30
if not REPLAN:
    T = horizon
optimizer = shooting_optimizer(
    env.dynamics_fn, main_car.cost_fn, udim, horizon, env.dt, replan=REPLAN, T=T
)
runner = Runner(env, main_car)

y0_idx, y1_idx = 1, 5
state = copy.deepcopy(env.state)
# state[y0_idx] = 0.4
# state[y1_idx] = -0.2
state[y0_idx] = -0.5
state[y1_idx] = 0.05
# state[y0_idx] = 0.5
# state[y1_idx] = 0.0
env.state = state

actions = optimizer(env.state)
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
