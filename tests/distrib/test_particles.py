from rdb.distrib.particles import ParticleServer
from rdb.optim.mpc import build_mpc
from rdb.exps.utils import *
from rdb.infer import *
from jax import random
import numpyro.distributions as dist
import yaml, argparse, os
import numpy as onp
import itertools
import ptyest
import copy
import ray


rng_key = random.PRNGKey(random_key)


def env_fn():
    import gym, rdb.envs.drive2d

    env = gym.make(ENV_NAME)
    env.reset()
    return env


def controller_fn(env, name=""):
    controller, runner = build_mpc(
        env, env.main_car.cost_runtime, HORIZON, env.dt, replan=False, name=name
    )
    return controller, runner


design_data = load_params(join(examples_dir(), f"designs/{ENV_NAME}.yaml"))
assert design_data["ENV_NAME"] == ENV_NAME, "Environment name mismatch"

## Prior sampling & likelihood functions for PGM
prior = LogUniformPrior(
    rng_key=None,
    normalized_key=NORMALIZED_KEY,
    feature_keys=FEATURE_KEYS,
    log_max=MAX_WEIGHT,
)
ird_proposal = IndGaussianProposal(
    rng_key=None,
    normalized_key=NORMALIZED_KEY,
    feature_keys=FEATURE_KEYS,
    proposal_var=IRD_PROPOSAL_VAR,
)
designer_proposal = IndGaussianProposal(
    rng_key=None,
    normalized_key=NORMALIZED_KEY,
    feature_keys=FEATURE_KEYS,
    proposal_var=DESIGNER_PROPOSAL_VAR,
)

# Evaluation Server
eval_server = ParticleServer(env_fn, controller_fn, num_workers=4)

env = env_fn()


def make_tasks(num_tasks):
    return onp.random.choice(env.all_tasks, num_tasks)


def make_weights(num_weights):
    weights = []
    for _ in range(num_weights):
        w = {}
        for key in env.feature_keys:
            w[key] = onp.random.random(10)
        weights.append(w)
    return weights


controller, runner = build_mpc(env)


@pytest.parametrize.mark(
    "num_weights", "num_tasks", itertools.product([1, 5, 10], [1, 2, 5, 10, 20, 200])
)
def test_compute_tasks(num_weights, num_tasks):
    tasks = make_tasks(num_tasks)
    belief = Particle(
        rng_key=rng_key,
        env_fn=env_fn,
        controller=controller,
        runner=runner,
        normalized_key="dist_cars",
        weights=make_weights(num_weights),
    )
    belief = belief.subsample(self._num_active_sample)
    self._eval_server.compute_tasks(belief, candidates, verbose=True)
    self._eval_server.compute_tasks(all_obs[-1], candidates, verbose=True)
