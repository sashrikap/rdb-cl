"""Use IRD style sampling to find plausible weights.

Notes:
    * Based On Inverse Reward Design

"""
import gym, time, copy
import jax.numpy as np
import rdb.envs.drive2d

from rdb.optim.mpc import shooting_optimizer
from rdb.optim.runner import Runner
from rdb.visualize.render import render_env
from rdb.visualize.preprocess import normalize_features

env = gym.make("Week3_02-v0")
env.reset()

cost_runtime = env.main_car.cost_runtime
optimizer = shooting_optimizer(
    env.dynamics_fn, cost_runtime, env.udim, env.horizon, env.dt, replan=REPLAN, T=T
)
runner = Runner(env, cost_runtime=cost_runtime)
