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


def assert_equal(dicta, dictb):
    for key in dicta.keys():
        assert onp.allclose(dicta[key], dictb[key])


def test_random_probs():
    key = random.PRNGKey(0)
    probs = onp.ones(3) / 3
    arr = [1, 2, 3]
    results = []
    for _ in range(1000):
        results.append(random_choice(key, arr, 100, probs, replacement=True))
    mean = onp.array(results).mean()
    # TODO: Rough test, find better ways
    assert mean > 1.95 and mean < 2.05

    probs = onp.array([0.6, 0.2, 0.2])
    arr = [1, 2, 3]
    results = []
    for _ in range(1000):
        results.append(random_choice(key, arr, 4, probs, replacement=True))
    mean = onp.array(results).mean()
    # TODO: Rough test, find better ways
    assert mean >= 1.5 and mean <= 1.6


def test_random_speed():
    key = random.PRNGKey(0)
    probs = onp.random.random(500)
    arr = onp.random.random(500)
    results = []
    t1 = time()
    for _ in range(10):
        with Profiler("Random choice"):
            res = random_choice(key, onp.arange(500), 500, probs, replacement=True)
            assert len(res) == 500
    print(f"Compute 10x 500 random took {time() - t1:.3f}")


def test_cross_product():
    arr_a = np.array([1, 2, 3])
    arr_b = np.array([4, 5, 6, 7])
    cross_a, cross_b = cross_product(arr_a, arr_b, np.array, np.array)
    na, nb = len(arr_a), len(arr_b)
    assert np.allclose(cross_a, np.repeat(arr_a, nb))
    assert np.allclose(cross_b, np.tile(arr_b, na))


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


@pytest.mark.parametrize("num_weights", [1, 5, 10, 20])
def test_collect_trajs(num_weights):
    key = random.PRNGKey(0)
    tasks = random_choice(key, env.all_tasks, num_weights)
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
    weights_arr = weights.prepare(env.features_keys).numpy_array()

    last_actions, last_costs, last_feats, last_feats_sum, last_vios = (
        None,
        None,
        None,
        None,
        None,
    )

    for max_batch in [1, 2, 8]:
        actions, costs, feats, feats_sum, vios = collect_trajs(
            weights_arr, states, optimizer, runner, max_batch=max_batch
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
        if last_actions is not None:
            assert np.allclose(actions, last_actions)
            assert np.allclose(costs, last_costs)
            assert_equal(feats, last_feats)
            assert_equal(feats_sum, last_feats_sum)
            assert_equal(vios, last_vios)
        last_actions = actions
        last_costs = costs
        last_feats = feats
        last_feats_sum = feats_sum
        last_vios = vios
