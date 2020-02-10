import gym
import pytest
import numpy as onp
import rdb.envs.drive2d
from time import time
from jax import random
from numpyro.handlers import seed
from rdb.infer.utils import *
from rdb.optim.utils import *
from rdb.exps.utils import Profiler
from rdb.optim.mpc import build_mpc
from rdb.optim.runner import Runner


def test_random_probs():
    key = random.PRNGKey(0)
    random_choice_fn = seed(random_choice, key)
    probs = onp.ones(3) / 3
    arr = [1, 2, 3]
    results = []
    for _ in range(1000):
        results.append(random_choice_fn(arr, 100, probs, replacement=True))
    mean = onp.array(results).mean()
    # TODO: Rough test, find better ways
    assert mean > 1.95 and mean < 2.05

    probs = onp.array([0.6, 0.2, 0.2])
    arr = [1, 2, 3]
    results = []
    for _ in range(1000):
        results.append(random_choice_fn(arr, 4, probs, replacement=True))
    mean = onp.array(results).mean()
    # TODO: Rough test, find better ways
    assert mean > 1.55 and mean < 1.65


def test_random_speed():
    key = random.PRNGKey(0)
    random_choice_fn = seed(random_choice, key)
    probs = onp.random.random(500)
    arr = onp.random.random(500)
    results = []
    t1 = time()
    for _ in range(10):
        with Profiler("Random choice"):
            res = random_choice_fn(onp.arange(500), 500, probs, replacement=True)
            assert len(res) == 500
    print(f"Compute 10x 500 random took {time() - t1:.3f}")


env = gym.make("Week6_02-v1")  # Two Blockway
env.reset()
main_car = env.main_car
T = 10
optimizer, runner = build_mpc(
    env,
    main_car.cost_runtime,
    T,
    env.dt,
    replan=False,
    T=10,
    engine="jax",
    method="adam",
)


@pytest.mark.parametrize("num_weights", [1, 5, 10])
def test_collect_trajs(num_weights):
    key = random.PRNGKey(0)
    random_choice_fn = seed(random_choice, key)
    tasks = random_choice_fn(env.all_tasks, num_weights)
    states = env.get_init_states(tasks)

    weights = []
    for _ in range(num_weights):
        w = {}
        for key in env.features_keys:
            w[key] = onp.random.random()
        weights.append(w)
    weights = DictList(weights)

    nfeatures = len(env.features_keys)
    nvios = len(env.constraints_keys)

    actions, costs, feats, feats_sum, vios = collect_trajs(
        weights, states, optimizer, runner
    )
    udim = 2
    assert actions.shape == (num_weights, T, udim)
    assert costs.shape == (num_weights,)
    assert feats.shape == (num_weights, T)
    assert feats.num_keys == nfeatures
    assert feats_sum.shape == (num_weights,)
    assert feats_sum.num_keys == nfeatures
    assert vios.shape == (num_weights, T)
    assert vios.num_keys == nvios
