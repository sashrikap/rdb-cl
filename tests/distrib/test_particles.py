from rdb.distrib.particles import ParticleServer
from rdb.infer.particles import Particles
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

PARAMS = load_params("examples/params/active_template.yaml")
locals().update(PARAMS)


def env_fn(env_name=None):
    import gym, rdb.envs.drive2d

    if env_name is None:
        env_name = ENV_NAME
    env = gym.make(env_name)
    env.reset()
    return env


def controller_fn(env, name=""):
    controller, runner = build_mpc(
        env,
        env.main_car.cost_runtime,
        dt=env.dt,
        name=name,
        test_mode=True,
        **IRD_CONTROLLER_ARGS
    )
    return controller, runner


eval_server = ParticleServer(
    env_fn,
    controller_fn,
    parallel=EVAL_ARGS["parallel"],
    normalized_key=WEIGHT_PARAMS["normalized_key"],
    weight_params=WEIGHT_PARAMS,
    max_batch=EVAL_ARGS["max_batch"],
)
eval_server.register("Evaluation", EVAL_ARGS["num_eval_workers"])


def make_tasks(num_tasks):
    idxs = onp.random.choice(onp.arange(len(env.all_tasks)), num_tasks)
    return env.all_tasks[idxs]


def make_weights(num_weights):
    weights = []
    for _ in range(num_weights):
        w = {}
        for key in env.features_keys:
            w[key] = onp.random.random()
        weights.append(w)
    return DictList(weights)


env = env_fn(ENV_NAME)
controller, runner = controller_fn(env)
rng_key = random.PRNGKey(1)

# @pytest.mark.parametrize("num_weights", [1, 5, 10])
# @pytest.mark.parametrize("num_tasks", [1, 2, 5, 10, 20, 200])
def test_compute_tasks(num_weights, num_tasks):
    tasks = make_tasks(num_tasks)
    belief = Particles(
        rng_name="test",
        rng_key=rng_key,
        env_fn=env_fn,
        controller=controller,
        runner=runner,
        save_name="save",
        normalized_key="dist_cars",
        weights=make_weights(num_weights),
    )
    eval_server.compute_tasks("Evaluation", belief, tasks, verbose=True)
    feats_sum = belief.get_features_sum(tasks)
    assert feats_sum.shape == (num_tasks, num_weights)
    costs = belief.get_costs(tasks)
    assert costs.shape == (num_tasks, num_weights)


test_compute_tasks(5, 10)
