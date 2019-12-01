import gym
import time
import copy
import jax.numpy as np
import numpy as onp
import rdb.envs.drive2d
import rdb.optim.mpc as mpc
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
opt_u_fn = mpc.optimize_u_fn(env.dynamics_fn, main_car.cost_fn, udim, horizon, env.dt)

now = time.time()
state0 = copy.deepcopy(env.state)
y0_idx = 1
y1_idx = 5
# y0_idx = 5
# y1_idx = 1


opt_u, r_max, info = opt_u_fn(np.zeros((horizon, udim)) * 0.1, env.state)
y0_range = [0.2]
y1_range = np.arange(-1.5, 1.51, 0.05)
num0 = len(y0_range)
num1 = len(y1_range)

best_rews = []
best_rews_std = []
best_acts = []
less_rews = []
cost_fns = []
for y0 in y0_range:
    for y1 in tqdm(y1_range):
        state = copy.deepcopy(state0)
        state[y0_idx] = y0
        state[y1_idx] = y1
        opt_u, c_min, info = opt_u_fn(np.zeros((horizon, udim)) * 0.1, state)
        r_max = -1.0 * c_min
        rews = -1.0 * info["costs"]
        best_rews.append(r_max)
        best_rews_std.append(np.std(rews))
        best_acts.append(opt_u)
        cost_fns.append(info["cost_fn"])

best_rews = np.array(best_rews).reshape((num0, num1))
best_acts = np.array(best_acts).reshape((num0, num1, horizon, udim))
best_rews_std = np.array(best_rews_std).reshape((num0, num1))

for y0, best_acts_y0 in zip(y0_range, best_acts):
    for y1, cost_fn in tqdm(zip(y1_range, cost_fns)):
        state = copy.deepcopy(state0)
        state[y0_idx] = y0
        state[y1_idx] = y1
        for opt_u in best_acts_y0:
            c_min = cost_fn(opt_u)
            less_rews.append(-1 * c_min)

less_rews = np.array(less_rews).reshape((num0, num1, num1))

fig = plt.figure()
ax = plt.axes()
# for less_rews_act in less_rews[0].T:
#    plt.plot(y1_range, less_rews_act)
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
