import gym
import time, copy
import jax.numpy as np
import rdb.envs.drive2d

from rdb.optim.mpc import shooting_method
from rdb.optim.runner import Runner
from rdb.visualize.render import render_env
from rdb.visualize.preprocess import normalize_features

DUMMY_ACTION = False
DRAW_HEAT = False
REPLAN = False
MAKE_MP4 = False

env = gym.make("Week3_02-v0")
obs = env.reset()
main_car = env.main_car
udim = 2
horizon = 10

T = 30
# T = 10
weights = {
    "dist_cars": 100.0,
    "dist_lanes": 50.0,
    "dist_fences": 300.0,
    "speed": 20.0,
    "control": 80.0,
}

if not DUMMY_ACTION:
    if not REPLAN:
        T = horizon
    optimizer, runner = shooting_method(
        env, main_car.cost_runtime, udim, horizon, env.dt, replan=REPLAN, T=T
    )

    y0_idx, y1_idx = 1, 5
    state = copy.deepcopy(env.state)
    state[y0_idx] = 0.2
    state[y1_idx] = -0.7
    env.set_task(state)

    env.reset()
    actions = optimizer(env.state, weights=weights)
    traj, cost, info = runner(env.state, actions, weights=weights)
    print(f"Total cost {cost}")
    violations = info["violations"]
    for k, v in violations.items():
        print(f"Violations {k}: {v.sum()}")
else:
    actions = np.zeros((T, udim))

env.reset()
env.render("human", draw_heat=DRAW_HEAT)

for t in range(T):
    env.step(actions[t])
    env.render("human", draw_heat=DRAW_HEAT)
    time.sleep(0.2)

if MAKE_MP4:
    pathname = f"data/y0({state[y0_idx]:.2f})_y1({state[y1_idx]:.2f}) theta 2.mp4"
    render_env(env, state, actions, 10, pathname)
