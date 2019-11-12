import gym
import time
import copy
import jax.numpy as np
import numpy as onp
import rdb.envs.drive2d
from rdb.optim.open import shooting_optimizer
from rdb.optim.runner import Runner
from tqdm import tqdm
from rdb.visualize.render import render_env
from rdb.visualize.plot import plot_3d

MAKE_MP4 = True
SAVE_PNG = True
REPLAN = True

env = gym.make("Week3_01-v0")
obs = env.reset()
main_car = env.main_car
udim = 2
horizon = 10
T = 30
optimizer = shooting_optimizer(
    env.dynamics_fn, main_car.cost_runtime, udim, horizon, env.dt, replan=REPLAN, T=T
)
runner = Runner(env, main_car)

state0 = copy.deepcopy(env.state)
y0_idx = 1
y1_idx = 5

weights = {
    "dist_cars": 100.0,
    "dist_lanes": 10.0,
    "dist_fences": 300.0,
    "speed": 20.0,
    "control": 80.0,
}

# y0_range = np.arange(-1.5, 1.5, 0.05)
# y1_range = np.arange(-1.5, 1.5, 0.05)
# y0_range = np.arange(-1.0, 1.01, 0.1)
# y1_range = np.arange(-1.0, 1.01, 0.1)
# y0_range = np.arange(-1.0, 1.01, 0.9)
# y1_range = np.arange(-1.0, 1.01, 0.9)
# y0_range = np.arange(-0.5, 0.51, 0.1)
# y1_range = np.arange(-0.5, 0.51, 0.1)
y0_range = np.arange(-0.5, 0.51, 0.5)
y1_range = np.arange(-0.5, 0.51, 0.5)

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
        actions = optimizer(state, weights=weights)
        traj, cost, info = runner(state, actions)
        rew = -1 * cost

        rews.append(rew)
        trajs.append(traj)
        list_rews.append(rew)
        list_actions.append(actions)
        list_states.append(state)
        list_paths.append(f"data/191111/y0({y0:.2f})_y1({y1:.2f}).mp4")
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
        render_env(env, state, actions, fps=3, path=path, savepng=SAVE_PNG)


plot_3d(y0_range, y1_range, all_rews)
