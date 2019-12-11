import gym
import time
import copy
import jax.numpy as np
import numpy as onp
import rdb.envs.drive2d
from rdb.optim.mpc import shooting_method
from rdb.optim.runner import Runner
from tqdm.auto import tqdm
from rdb.visualize.render import render_env
from rdb.visualize.plot import plot_3d

MAKE_MP4 = False
SAVE_PNG = False
REPLAN = False
VERBOSE = False

env = gym.make("Week3_02-v0")
obs = env.reset()
main_car = env.main_car
horizon = 10
T = 10
controller, runner = shooting_method(
    env, main_car.cost_runtime, horizon, env.dt, replan=REPLAN, T=T
)

state0 = copy.deepcopy(env.state)
y0_idx = 1
y1_idx = 5

"""weights = {
    "dist_cars": 100.0,
    "dist_lanes": 50.0,
    "dist_fences": 300.0,
    "speed": 20.0,
    "control": 80.0,
}"""
weights = {
    "dist_cars": 100.0,
    "dist_lanes": 13.2,
    "dist_fences": 233.0,
    "speed": 160.0,
    "control": 31.0,
}

y0_range = np.arange(-0.5, 0.51, 0.1)
y1_range = np.arange(-0.5, 0.51, 0.1)

list_actions = []
list_costs = []
list_trajs = []
list_init_states = []
list_violations = []
highest_violation = 0.0
highest_state = None
for y0 in tqdm(y0_range):
    for y1 in y1_range:
        state = copy.deepcopy(state0)
        state[y0_idx] = y0
        state[y1_idx] = y1
        env.set_task(state)
        list_init_states.append(state)

        actions = controller(env.state, weights=weights)
        traj, cost, info = runner(env.state, actions, weights=weights)
        violations = info["violations"]

        list_trajs.append(traj)
        list_costs.append(cost)
        list_actions.append(actions)
        num_violate = sum([sum(v) for v in violations.values()])
        if num_violate > highest_violation:
            highest_violation = num_violate
            highest_state = state
            print(f"y0 {y0:.2f} y1 {y1:.2f} violations {num_violate}")
        list_violations.append(num_violate)
        if VERBOSE:
            print(f"y0 {y0:.2f} y1 {y1:.2f} violations {num_violate}")

all_violations = onp.array(list_violations).reshape((len(y0_range), len(y1_range)))
print(f"Overall mean violations {all_violations.mean():.3f}")
plot_3d(y0_range, y1_range, all_violations)

if MAKE_MP4:
    for actions, cost, state in tqdm(
        zip(list_actions, list_costs, list_init_states), total=len(list_actions)
    ):
        y0, y1 = state[y0_idx], state[y1_idx]
        env.set_task(state)
        env.reset()
        text = f"Total cost: {-1 * cost:.3f}\nTesting\ny0({y0:.2f}) y1({y1:.2f})"
        mp4_path = f"data/191208/recording_y0({y0:.2f})_y1({y1:.2f}).mp4"
        runner.collect_mp4(env, state, actions, fps=3, path=path, text=text)
