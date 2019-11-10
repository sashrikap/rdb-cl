import gym
import time, copy
import jax.numpy as np
import rdb.envs.drive2d
from rdb.optim.open import shooting_optimizer
from matplotlib import pyplot as plt
from rdb.visualize.render import render_env

REOPTIMIZE = True
MAKE_MP4 = True

env = gym.make("Week3_01-v0")
obs = env.reset()
main_car = env.main_car
udim = 2
horizon = 10

T = 30
if not REOPTIMIZE:
    T = horizon
optimizer = shooting_optimizer(
    env.dynamics_fn, main_car.cost_fn, udim, horizon, env.dt, mpc=REOPTIMIZE, T=T
)

y0_idx, y1_idx = 1, 5
state = copy.deepcopy(env.state)
# state[y0_idx] = 0.4
# state[y1_idx] = -0.2
state[y0_idx] = -0.5
state[y1_idx] = 0.05
# state[y0_idx] = 0.5
# state[y1_idx] = 0.0
env.state = state

opt_u, c_min, info = optimizer(np.zeros((horizon, udim)), env.state)
r_max = -1 * c_min
env.render("human")
print(f"Rmax {r_max}")
print(opt_u)


total_cost = 0
actions = []
for t in range(T):
    action = opt_u[t]
    actions.append(action)
    cost = main_car.cost_fn(env.state, action)
    total_cost += cost
    env.step(action)
    env.render("human")

    r_max = -1 * c_min
    time.sleep(0.2)

print(f"Rew {-1 * total_cost:.3f}")

if MAKE_MP4:
    pathname = f"data/y0({state[y0_idx]:.2f})_y1({state[y1_idx]:.2f}) theta 2.mp4"
    render_env(env, state, actions, 10, pathname)
