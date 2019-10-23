import gym
import time
import copy
import jax.numpy as np
import numpy as onp
import rdb.envs.drive2d
import rdb.optim.open as opt_open
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import cm
from rdb.visualize.render import *


env = gym.make("Week3_01-v0")
obs = env.reset()
main_car = env.main_car
udim = 2
horizon = 15
opt_u_fn = opt_open.optimize_u_fn(
    env.dynamics_fn, main_car.cost_fn, udim, horizon, env.dt
)

now = time.time()
state0 = copy.deepcopy(env.state)
# y0_idx = 1
# y1_idx = 5
y0_idx = 5
y1_idx = 1


opt_u, r_max, info = opt_u_fn(np.ones((horizon, udim)) * 0.1, env.state)
y0_range = [0.3]
y1_range = np.arange(-1.00, 1.50, 0.05)

all_rews = []
all_cross_rews = []
list_cost_fns = []
for y0 in y0_range:
    rews = []
    list_opt_u = []
    for y1 in tqdm(y1_range):
        state = copy.deepcopy(state0)
        state[y0_idx] = y0
        state[y1_idx] = y1
        opt_u, c_min, info = opt_u_fn(np.ones((horizon, udim)) * 0.1, state)
        r_max = -1 * c_min
        rews.append(r_max)
        list_opt_u.append(opt_u)
        list_cost_fns.append(info["cost_fn"])
    all_rews.append(rews)
    for y1, cost_fn in tqdm(zip(y1_range, list_cost_fns)):
        state = copy.deepcopy(state0)
        state[y0_idx] = y0
        state[y1_idx] = y1
        cross_rews = []
        for opt_u in list_opt_u:
            c_min = cost_fn(opt_u)
            cross_rews.append(-1 * c_min)
        all_cross_rews.append(cross_rews)


fig = plt.figure()
ax = plt.axes()
for cross_rew, y1 in zip(all_cross_rews, y1_range):
    plt.plot(y1_range, cross_rew)
plt.plot(
    y1_range,
    all_rews[0],
    label="l-bfgs",
    marker="",
    color="orange",
    linewidth=4,
    alpha=0.7,
)
ax.set_xlabel("Car position")
ax.set_ylabel("Reward")
ax.set_title(f"Cross reward comparison (Car 0 at {y0_range[0]:.3f})")
ax.legend()
plt.show()
