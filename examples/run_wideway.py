import gym
import time, copy
import jax.numpy as np
import rdb.envs.drive2d

from rdb.optim.runner import Runner
from rdb.optim.mpc import build_mpc
from rdb.visualize.render import render_env
from rdb.visualize.preprocess import normalize_features

DRAW_HEAT = False
DUMMY_ACTION = False
COLLECT_MP4 = False

env = gym.make("Week5_01-v0")
obs = env.reset()
horizon = 10

T = 30
main_car = env.main_car

if not DUMMY_ACTION:
    optimizer, runner = build_mpc(env, main_car.cost_runtime, horizon, env.dt, T=T)

    weights = {
        "dist_cars": 1.0,
        "dist_lanes": 0.0,
        "dist_fences": 0.0,
        "dist_entrance": 10.0,
        "dist_garage": 10.0,
        "speed": 0.0,
        "control": 0.4,
    }

    actions = optimizer(env.state, weights=weights)
    traj, cost, info = runner(env.state, actions, weights=weights)
    print(f"Total cost {cost}")
    if COLLECT_MP4:
        runner.collect_mp4(env.state, actions, path="data/driveway_sanity.mp4")
else:
    actions = np.zeros((T, env.udim))

env.render("human", draw_heat=DRAW_HEAT)
for t in range(T):
    env.step(actions[t])
    env.render("human", draw_heat=DRAW_HEAT)
    time.sleep(0.2)
