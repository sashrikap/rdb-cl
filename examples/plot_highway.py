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
horizon = 10
opt_u_fn = opt_open.optimize_u_fn(
    env.dynamics_fn, main_car.cost_fn, udim, horizon, env.dt
)

now = time.time()
state0 = copy.deepcopy(env.state)
y0_idx = 1
y1_idx = 5


opt_u, r_max, info = opt_u_fn(np.zeros((horizon, udim)) * 0.1, env.state)
y0_range = np.arange(-1.5, 1.5, 0.05)
y1_range = np.arange(-1.5, 1.5, 0.05)

all_rews = []
list_opt_u = []
list_states = []
list_rews = []
list_paths = []
for y0 in tqdm(y0_range):
    rews = []
    for y1 in y1_range:
        state = copy.deepcopy(state0)
        state[y0_idx] = y0
        state[y1_idx] = y1
        opt_u, c_min, info = opt_u_fn(np.zeros((horizon, udim)) * 0.1, state)
        r_max = -1 * c_min
        rews.append(r_max)
        list_rews.append(r_max)
        list_opt_u.append(opt_u)
        list_states.append(state)
        list_paths.append(f"data/plot/y0({y0:.2f})_y1({y1:.2f})_rew({r_max:.3f}).mp4")
    all_rews.append(rews)

# batch_render_env(env, list_states, list_opt_u, list_paths, fps=3, width=300)
"""for state, opt_u, path in zip(list_states, list_opt_u, list_paths):
    print(f"state {np.mean(state)} opt u {np.mean(opt_u)}")
    render_env(env, state, opt_u, fps=3, path=path)"""


fig = plt.figure()
ax = plt.axes(projection="3d")
X, Y = onp.meshgrid(y0_range, y1_range)
Z = onp.array(all_rews).T
# ax.plot_wireframe(X, Y, Z, color='black')
surf = ax.plot_surface(
    X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True, rstride=1, cstride=1
)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel("Car 0 position")
ax.set_ylabel("Car 1 position")
ax.set_title("Reward")
plt.show()
