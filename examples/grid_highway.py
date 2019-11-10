import gym
import time
import copy
import jax.numpy as np
import numpy as onp
import rdb.envs.drive2d
import rdb.optim.open as opt_open
from tqdm import tqdm
from rdb.visualize.render import *

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm

MAKE_MP4 = True

env = gym.make("Week3_01-v0")
obs = env.reset()
main_car = env.main_car
udim = 2
horizon = 10
T = 30
opt_u_fn = opt_open.optimize_u_fn(
    env.dynamics_fn, main_car.cost_fn, udim, horizon, env.dt
)

now = time.time()
state0 = copy.deepcopy(env.state)
y0_idx = 1
y1_idx = 5


# y0_range = np.arange(-1.5, 1.5, 0.05)
# y1_range = np.arange(-1.5, 1.5, 0.05)
# y0_range = np.arange(-1.0, 1.01, 0.1)
# y1_range = np.arange(-1.0, 1.01, 0.1)
# y0_range = np.arange(-1.0, 1.01, 0.9)
# y1_range = np.arange(-1.0, 1.01, 0.9)
y0_range = np.arange(-0.5, 0.51, 0.1)
y1_range = np.arange(-0.5, 0.51, 0.1)

all_rews = []
all_trajs = []
list_opt_u = []
list_states = []
list_rews = []
list_paths = []
for y0 in tqdm(y0_range):
    rews = []
    trajs = []
    for y1 in y1_range:
        state = copy.deepcopy(state0)
        state[y0_idx] = y0
        state[y1_idx] = y1
        opt_u, c_min, info = opt_u_fn(np.zeros((horizon, udim)), state, mpc=True, T=T)
        r_max = -1 * c_min
        rews.append(r_max)
        trajs.append(info["xs"])
        list_rews.append(r_max)
        list_opt_u.append(opt_u)
        list_states.append(state)
        list_paths.append(f"data/191030/y0({y0:.2f})_y1({y1:.2f})_rew({r_max:.3f}).mp4")
        # list_paths.append(f"data/191101/y0({y0:.2f})_y1({y1:.2f})_rew({r_max:.3f}).mp4")
    all_trajs.append(trajs)
    all_rews.append(rews)


list_states, list_opt_u, list_paths = (
    onp.array(list_states),
    onp.array(list_opt_u),
    onp.array(list_paths),
)
list_rews = onp.array(all_rews).flatten()
ind = onp.argsort(list_rews, axis=0)
list_states, list_opt_u, list_paths = list_states[ind], list_opt_u[ind], list_paths[ind]
if MAKE_MP4:
    for state, opt_u, path in tqdm(
        zip(list_states, list_opt_u, list_paths), total=len(list_states)
    ):
        # print(f"state {np.mean(state)} opt u {np.mean(opt_u)}")
        render_env(env, state, opt_u, fps=3, path=path)


fig = plt.figure()
ax = plt.axes(projection="3d")
X, Y = onp.meshgrid(y0_range, y1_range)
Z = onp.array(all_rews).T

# onp.savez("grid.npz", y0_range, y1_range, all_rews)
# ax.plot_wireframe(X, Y, Z, color='black')
surf = ax.plot_surface(
    X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True, rstride=1, cstride=1
)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel("Car 0 position")
ax.set_ylabel("Car 1 position")
ax.set_title("Reward")
plt.show()
