"""Costs are deducted with max_costs for normalization.

This script samples different weights, and compare the resulting trajectory against max cost.
"""


import gym
import copy
import pytest
import jax.numpy as np
import numpy as onp
import rdb.envs.drive2d
from time import time
from jax.config import config
from rdb.optim.mpc import *
from rdb.optim.runner import Runner

config.update("jax_enable_x64", True)


ENV_NAME = "Week6_02-v1"  # Two Blockway
env = gym.make(ENV_NAME)
env.reset()
main_car = env.main_car
horizon = 10
T = 10
xdim, udim = env.state.shape[1], 2
true_weights = {
    "dist_cars": 1.1,
    "dist_lanes": 0.1,
    "dist_fences": 0.35,
    "dist_objects": 1.25,
    "speed": 0.05,
    "control": 0.1,
}
normal_key = "dist_cars"


def make_weights(num_weights):
    keys = ["dist_lanes", "dist_fences", "dist_objects", "speed", "control"]
    weights = []
    for _ in range(num_weights):
        w = {normal_key: 1.0}
        for key in keys:
            w[key] = onp.exp(onp.random.random() * 8 - 4)
        weights.append(w)
    return DictList(weights)


def get_optimizer():
    optimizer, runner = build_mpc(
        env,
        main_car.cost_runtime,
        horizon,
        env.dt,
        replan=5,
        T=T,
        engine="jax",
        method="adam",
    )
    return optimizer, runner


def get_init_states(nbatch):
    num_tasks = len(env.all_tasks)
    task = env.all_tasks[onp.random.randint(0, num_tasks, size=1)]
    states = onp.concatenate([env.get_init_state(task[0])] * nbatch, axis=0)
    return task, states


def run_single(states):
    max_feats = DictList([env.max_feats_dict])
    # import pdb; pdb.set_trace()
    state = states[0, None, :]
    acs = optimizer_one(state, weights=true_weights, batch=False)
    _, _, info = runner_one(state, acs, weights=true_weights, batch=False)
    weights_dict = (
        DictList([true_weights]).prepare(env.features_keys).normalize_by_key(normal_key)
    )
    feats_sum = info["feats_sum"]
    costs_to_max = (weights_dict * (feats_sum - max_feats)).onp_array().mean(axis=0)
    return costs_to_max


def run_batch(states):
    max_feats = DictList([env.max_feats_dict])
    acs_all = optimizer_batch(states, weights=weights)
    traj_all, costs_all, info_all = runner_batch(
        states, acs_all, weights=weights, batch=True
    )
    weights_dict = (
        DictList([true_weights]).prepare(env.features_keys).normalize_by_key(normal_key)
    )
    feats_sum = info_all["feats_sum"]
    costs_all = (weights_dict * (feats_sum - max_feats)).onp_array().mean(axis=0)
    return costs_all


def tune_normal_cost(nbatch):
    """Compare method {engine, method} vs single l-bfgs"""

    task, states = get_init_states(nbatch)
    cost_one = run_single(states)
    costs_batch = run_batch(states)

    ## Only check if batch is worse than single
    regrets = costs_batch - cost_one
    max_rew = -1 * cost_one
    ratios = regrets / max_rew
    print(f"True rew {(-1 * float(cost_one)):.02f}")
    print(
        f"Ratios mean {np.mean(ratios):.02f} std {np.std(ratios):.02f} max {np.max(ratios):.02f} min {np.min(ratios):.02f}"
    )
    if np.max(ratios) > 1:
        import pdb

        pdb.set_trace()
    print(f"Better than {100 * (regrets > 0).sum() / float(nbatch)} percent")


if __name__ == "__main__":

    optimizer_one, runner_one = get_optimizer()
    optimizer_batch, runner_batch = get_optimizer()

    ntasks = 20
    nbatch = 100
    for _ in range(ntasks):
        weights = make_weights(nbatch)
        tune_normal_cost(nbatch)
