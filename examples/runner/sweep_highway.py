import gym
import time
import copy
import jax.numpy as np
import numpy as onp
import rdb.envs.drive2d
from rdb.optim.mpc import shooting_optimizer
from rdb.optim.runner import Runner
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import cm
from rdb.visualize.render import *


env = gym.make("Week3_01-v0")
obs = env.reset()
main_car = env.main_car
udim = 2
horizon = 10
optimizer = shooting_optimizer(env.dynamics_fn, main_car.cost_fn, udim, horizon, env.dt)
runner = Runner(env, main_car.cost_runtime, main_car.cost_fn)

state = copy.deepcopy(env.state)
y0_idx = 1
y1_idx = 5


actions = optimizer(env.state)
traj, cost, info = runner(env.state, actions)
y0_y1_pairs = [(0.4, -0.2), (-0.5, 0.05), (0.5, 0.0)]

best_rews = []
best_rews_std = []
best_acts = []
less_rews = []
for (y0, y1) in y0_y1_pairs:
    state = copy.deepcopy(state)
    state[y0_idx] = y0
    state[y1_idx] = y1
    actions = optimizer(state)
    traj, cost, info = runner(state, actions)
    rew = -1.0 * cost
    best_rews.append(rew)
    best_rews_std.append(np.std(info["costs"]))
    best_acts.append(actions)

best_rews = np.array(best_rews).reshape((num0, num1))
best_acts = np.array(best_acts).reshape((num0, num1, horizon, udim))
best_rews_std = np.array(best_rews_std).reshape((num0, num1))

fig = plt.figure()
ax = plt.axes()
plt.plot(
    y1_range,
    best_rews[0],
    label="l-bfgs",
    marker="",
    color="orange",
    linewidth=4,
    alpha=0.7,
)
plt.fill_between(
    y1_range,
    best_rews[0] - best_rews_std[0],
    best_rews[1] + best_rews_std[1],
    color="orange",
    alpha=0.4,
)
ax.set_xlabel("Car position")
ax.set_ylabel("Reward")
ax.set_title(f"Cross reward comparison (Car 0 at {y0_range[0]:.3f})")
ax.legend()
plt.show()
