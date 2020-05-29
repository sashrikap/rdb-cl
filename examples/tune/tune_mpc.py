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
weights = {
    "dist_cars": 1.1,
    "dist_lanes": 0.1,
    "dist_fences": 0.35,
    "dist_objects": 1.25,
    "speed": 0.05,
    "control": 0.1,
}
normal_key = "dist_cars"


def get_mpc(engine, method):
    optimizer, runner = build_mpc(
        env,
        main_car.cost_runtime,
        horizon,
        env.dt,
        replan=5,
        T=T,
        engine=engine,
        method=method,
    )
    return optimizer, runner


def get_init_states(nbatch):
    num_tasks = len(env.all_tasks)
    tasks = env.all_tasks[onp.random.randint(0, num_tasks, size=nbatch)]
    states = onp.concatenate([env.get_init_state(t) for t in tasks], axis=0)
    return states


def run_single(states, optimizer, runner, weights):
    costs_single = []
    for state_i in tqdm(states, desc=f"Running single optimzer"):
        state_i = state_i[None, :]
        acs_i = optimizer(state_i, weights=weights, batch=False)
        traj_i, cost_i, info_i = runner(state_i, acs_i, weights=weights, batch=False)
        weights_dict = (
            DictList([weights]).prepare(env.features_keys).normalize_by_key(normal_key)
        )
        max_feats = DictList([env.max_feats_dict])
        feats_sum = info_i["feats_sum"]
        costs_to_max = (weights_dict * (feats_sum - max_feats)).onp_array().mean(axis=0)
        costs_single.append(cost_i)
    costs_single = onp.concatenate(costs_single)
    return costs_single


def run_batch(states, optimizer, runner, weights):
    weights_all = [weights] * len(states)
    acs_all = optimizer(states, weights=weights_all)
    traj_all, costs_all, info_all = runner(
        states, acs_all, weights=weights_all, batch=True
    )
    return costs_all


def tune_method(nbatch, engine, method, engine_gt="scipy", method_gt="lbfgs"):
    """Compare method {engine, method} vs single l-bfgs"""
    opt_a, runner_a = get_mpc(engine, method)
    opt_gt, runner_gt = get_mpc(engine_gt, method_gt)
    states = get_init_states(nbatch)
    ## Single
    # t_single = time()
    costs_single = run_single(states, opt_gt, runner_gt, weights)
    # t_single = time() - time()
    ## Batch
    # t_batch = time()
    costs_batch = run_batch(states, opt_a, runner_a, weights)
    # t_batch = time() - t_batch
    print(f"Nbatch {nbatch}")
    ## Only check if batch is worse than single
    abs_diff = onp.maximum(costs_batch - costs_single, 0)
    abs_ratio = abs_diff / costs_single
    max_idx = abs_ratio.argmax()
    max_ratio = abs_ratio[max_idx]
    mean_ratio = abs_ratio.mean()
    median_ratio = np.median(abs_ratio)
    print(
        f"Batch {method} x {nbatch} vs single {method_gt} diff max ratio: {max_ratio:.3f} median: {median_ratio:.3f} mean: {mean_ratio:.3f}"
    )


if __name__ == "__main__":
    # tune_method(100, "scipy", "lbfgs")
    # tune_method(100, "scipy", "lbfgs", "jax", "adam")
    tune_method(500, "scipy", "lbfgs")
    # tune_method(500, "jax", "adam")
    # tune_method(100, "scipy", "bfgs")
    # tune_method(100, "scipy", "bfgs", engine_gt="scipy", method_gt="bfgs")
    # tune_method(100, "scipy", "basinhopping")
    # tune_method(100, "jax", "adam")
    # tune_method(1000, "jax", "adam")
    # tune_method(100, "jax", "momentum")
