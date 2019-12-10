import gym
import time
import copy
import jax.numpy as np
import numpy as onp
import rdb.envs.drive2d
from rdb.optim.mpc import shooting_method
from rdb.optim.runner import Runner
from tqdm import tqdm
from rdb.visualize.render import render_env
from rdb.visualize.plot import plot_3d

MAKE_MP4 = True
SAVE_PNG = True
REPLAN = True

env = gym.make("Week3_02-v0")
obs = env.reset()
main_car = env.main_car
udim = 2
horizon = 10
T = 30
optimizer, runner = shooting_method(
    env, main_car.cost_runtime, udim, horizon, env.dt, replan=REPLAN, T=T
)

state0 = copy.deepcopy(env.state)
y0_idx = 1
y1_idx = 5

weights = {
    "dist_cars": 40.0,
    "dist_lanes": 50.0,
    "dist_fences": 1.0,
    "speed": 100.0,
    "control": 8.0,
}

# y0_range = np.arange(-1.5, 1.5, 0.05)
# y1_range = np.arange(-1.5, 1.5, 0.05)
# y0_range = np.arange(-1.0, 1.01, 0.1)
# y1_range = np.arange(-1.0, 1.01, 0.1)
# y0_range = np.arange(-1.0, 1.01, 0.9)
# y1_range = np.arange(-1.0, 1.01, 0.9)
# y0_range = np.arange(-0.5, 0.51, 0.1)
# y1_range = np.arange(-0.5, 0.51, 0.1)
y0_range = np.arange(-0.5, 0.51, 0.1)
y1_range = np.arange(-0.5, 0.51, 0.1)

list_actions = []
list_costs = []
list_trajs = []
list_pairs = []
for y0 in tqdm(y0_range):
    for y1 in y1_range:
        env.set_task(y0, y1)
        actions = optimizer(env.state, weights=weights)
        traj, cost, info = runner(env.state, actions, weights=weights)

        list_trajs.append(traj)
        list_costs.apoend(cost)
        list_pairs.append((y0, y1))
        list_rews.append(rew)
        list_actions.append(actions)

list_actions = onp.array(list_actions)
list_rews = onp.array(list_rews)


if MAKE_MP4:
    for actions, rew, pair in tqdm(
        zip(list_actions, list_rews, list_pairs), total=len(list_pairs)
    ):
        y0, y1 = pair
        env.set_task(y0, y1)
        env.reset()
        state = env.state
        text = f"Total cost: {-1 * rew:.3f}\nTesting\ny0({y0:.2f}) y1({y1:.2f})"
        mp4_path = f"data/191113/testing/recording_y0({y0:.2f})_y1({y1:.2f}).mp4"
        tbn_path = f"data/191113/testing/thumbnail_y0({y0:.2f})_y1({y1:.2f}).png"
        runner.collect_mp4(
            env, state, actions, fps=3, path=path, savepng=SAVE_PNG, text=text
        )


plot_3d(y0_range, y1_range, all_rews)
