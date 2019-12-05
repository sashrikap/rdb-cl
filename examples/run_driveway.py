import gym
import time, copy
import jax.numpy as np
import rdb.envs.drive2d

from rdb.optim.runner import Runner
from rdb.visualize.render import render_env
from rdb.optim.mpc import shooting_optimizer
from rdb.visualize.preprocess import normalize_features

DRAW_HEAT = False
DUMMY_ACTION = False

env = gym.make("Week4_01-v0")
obs = env.reset()
udim = 2
horizon = 10

T = 30
main_car = env.main_car

if not DUMMY_ACTION:
    optimizer = shooting_optimizer(
        env.dynamics_fn, main_car.cost_runtime, udim, horizon, env.dt, T=T
    )
    runner = Runner(env, main_car.cost_runtime, main_car.cost_fn)

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
    traj, cost, info = runner(env.state, actions)
    print(f"Total cost {cost}")

else:
    actions = np.zeros((T, udim))

env.render("human", draw_heat=DRAW_HEAT)
for t in range(T):
    env.step(actions[t])
    env.render("human", draw_heat=DRAW_HEAT)
    time.sleep(0.2)