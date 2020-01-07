"""Compare a set of Active-IRD returned weights and true weights.
"""

import os
import gym
import jax.numpy as np
import rdb.envs.drive2d
from jax import random
from rdb.infer.particles import Particles
from rdb.optim.utils import multiply_dict_by_keys
from rdb.optim.mpc import shooting_method

w_true = {
    "dist_cars": 1.0,
    "dist_lanes": 0.1,
    "dist_fences": 0.6,
    "speed": 0.5,
    "control": 0.16,
}

w_ird = {
    "dist_cars": 1.0,
    "dist_lanes": 0.001,
    "dist_fences": 213.625,
    "speed": 2.842,
    "control": 0.208,
}

filepath = (
    "data/191219/active_ird_exp1/save/weights_seed_[ 0 17]_itr_08_method_infogain.npz"
)

key = random.PRNGKey(0)
HORIZON = 10


def env_fn():
    env = gym.make("Week3_02-v0")
    env.reset()
    return env


env = env_fn
controller, runner = shooting_method(
    env, env.main_car.cost_runtime, HORIZON, env.dt, replan=False
)
sample_ws = None
ps = Particles(key, env_fn, controller, runner, sample_ws)
ps.load(filepath)
ps.log_samples(1)

diffs = []
for ti, task in enumerate(env.all_tasks):
    env.set_task(task)
    env.reset()

    acs_true = controller(env.state, weights=w_true)
    traj_true, c_true, info_true = runner(env.state, acs_true, weights=w_true)
    featssum_true = info_true["feats_sum"]

    c_ird_dict = multiply_dict_by_keys(w_ird, featssum_true)
    c_ird = np.sum(list(c_ird_dict.values()))
    # print(f"True rew {c_true:.3f} / IRD rew {c_ird:.3f}")
    print(f"Task {ti} rew diff {c_true - c_ird:.3f}")
    diffs.append(c_true - c_ird)

diffs = np.array(diffs)
print(f"Min ratio {np.argmin(diffs)} ratio {min(diffs):.3f}")
