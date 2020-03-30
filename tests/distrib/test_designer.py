from rdb.distrib.designer import DesignerServer
from rdb.optim.mpc import build_mpc
from rdb.exps.utils import *
from rdb.infer import *
from jax import random
import numpyro.distributions as dist
import yaml, argparse, os
import numpy as onp
import itertools
import pytest
import copy
import ray


rng_key = random.PRNGKey(0)
ENV_NAME = "Week6_02-v1"


def env_fn():
    import gym, rdb.envs.drive2d

    env = gym.make(ENV_NAME)
    env.reset()
    return env


def designer_controller_fn(env, name=""):
    controller, runner = build_mpc(
        env,
        env.main_car.cost_runtime,
        dt=env.dt,
        replan=False,
        name=name,
        engine="jax",
        method="adam",
    )
    return controller, runner


weight_params = {
    "normalized_key": "dist_cars",
    "max_weights": 15.0,
    "bins": 200,
    "feature_keys": [
        "dist_cars",
        "dist_lanes",
        "dist_fences",
        "dist_objects",
        "speed",
        "control",
    ],
}


def prior_fn(name="", feature_keys=weight_params["feature_keys"]):
    return LogUniformPrior(
        normalized_key=weight_params["normalized_key"],
        feature_keys=feature_keys,
        log_max=weight_params["max_weights"],
        name=name,
    )


sample_init_args = {"proposal_var": 0.5, "max_val": 15.0}
sample_args = {"num_samples": 10, "num_warmup": 10, "num_chains": 1}

TRUE_W = {
    "control": 2,
    "dist_cars": 2.0,
    "dist_fences": 5.35,
    "dist_lanes": 2.5,
    "dist_objects": 4.25,
    "speed": 5,
    "speed_over": 80,
}


def designer_fn():
    designer = Designer(
        env_fn=env_fn,
        beta=10.0,
        controller_fn=designer_controller_fn,
        prior_fn=prior_fn,
        num_normalizers=1,
        weight_params=weight_params,
        normalized_key=weight_params["normalized_key"],
        save_root=f"test/test_distrib_designer",
        exp_name=f"test_distrib_designer",
        sample_init_args=sample_init_args,
        sample_args=sample_args,
    )
    designer.update_key(rng_key)
    designer.true_w = TRUE_W
    return designer


def run_server():
    designer_server = DesignerServer(designer_fn)
    N = 4
    env = env_fn()
    tasks = env.all_tasks[:N]
    designer_server.register(N)
    samples = designer_server.simulate(tasks, methods=["test"] * N, itr=0)
    for sp in samples:
        print("Sample weights shape", sp.weights.shape)
    # import pdb; pdb.set_trace()


run_server()
