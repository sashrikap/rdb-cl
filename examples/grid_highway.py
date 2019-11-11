import gym
import time
import copy
import jax.numpy as np
import numpy as onp
import rdb.envs.drive2d
from rdb.optim.open import shooting_optimizer
from rdb.optim.runner import Runner
from tqdm import tqdm
from rdb.visualize.render import *

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm

MAKE_MP4 = True
REPLAN = True

env = gym.make("Week3_01-v0")
obs = env.reset()
main_car = env.main_car
udim = 2
horizon = 10
T = 30
optimizer = shooting_optimizer(
    env.dynamics_fn, main_car.cost_fn, udim, horizon, env.dt, replan=REPLAN, T=T
)
runner = Runner(env, main_car)

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
list_actions = []
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
        actions = optimizer(state)
        traj, cost, info = runner(state, actions)
        rew = -1 * cost

        rews.append(rew)
        trajs.append(traj)
        list_rews.append(rew)
        list_actions.append(actions)
        list_states.append(state)
        list_paths.append(f"data/191030/y0({y0:.2f})_y1({y1:.2f})_rew({rew:.3f}).mp4")
        # list_paths.append(f"data/191101/y0({y0:.2f})_y1({y1:.2f})_rew({rew:.3f}).mp4")
    all_trajs.append(trajs)
    all_rews.append(rews)


list_states = onp.array(list_states)
list_actions = onp.array(list_actions)
list_paths = onp.array(list_paths)
list_rews = onp.array(all_rews).flatten()

ind = onp.argsort(list_rews, axis=0)
list_states, list_actions, list_paths = (
    list_states[ind],
    list_actions[ind],
    list_paths[ind],
)
if MAKE_MP4:
    for state, actions, path in tqdm(
        zip(list_states, list_actions, list_paths), total=len(list_states)
    ):
        render_env(env, state, actions, fps=3, path=path)


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
