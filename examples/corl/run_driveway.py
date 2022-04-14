import gym
import time, copy
import jax.numpy as jnp
import rdb.envs.drive2d

from rdb.optim.runner import Runner
from rdb.visualize.render import render_env
from rdb.optim.mpc import build_mpc
from rdb.visualize.preprocess import normalize_features

DRAW_HEAT = False
DUMMY_ACTION = False
COLLECT_MP4 = True

env = gym.make("Week4_01-v0")
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
    y0_idx, y1_idx = 1, 5
    state = copy.deepcopy(env.state)
    state[y0_idx] = -0.2
    state[y1_idx] = 0.0
    env.state = state

    actions = optimizer(env.state, weights=weights)
    traj, cost, info = runner(env.state, actions, weights=weights)
    print(f"Total cost {cost}")
    if COLLECT_MP4:
        runner.collect_mp4(env.state, actions, path="data/driveway_sanity3.mp4")
else:
    actions = jnp.zeros((T, env.udim))

env.render("human", draw_heat=DRAW_HEAT)
for t in range(T):
    env.step(actions[t])
    env.render("human", draw_heat=DRAW_HEAT)
    time.sleep(0.2)
