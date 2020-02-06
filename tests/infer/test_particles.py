import gym
import pytest
import numpy as onp
import rdb.envs.drive2d
from time import time
from jax import random
from numpyro.handlers import seed
from rdb.infer import *
from rdb.exps.utils import *
from rdb.optim.utils import *
from rdb.exps.utils import Profiler
from rdb.optim.mpc import build_mpc
from rdb.optim.runner import Runner

env = gym.make("Week6_02-v1")  # Two Blockway
env.reset()
normalized_key = "dist_cars"
T = 10


def build_weights(num_weights):
    weights = []
    for _ in range(num_weights):
        w = {}
        for key in env.features_keys:
            w[key] = onp.random.random()
        weights.append(w)
    return DictList(weights)


def build_particles(num_weights):
    env = gym.make("Week6_02-v1")  # Two Blockway
    env.reset()
    main_car = env.main_car
    controller, runner = build_mpc(
        env,
        main_car.cost_runtime,
        T,
        env.dt,
        replan=False,
        T=10,
        engine="jax",
        method="adam",
    )
    weights = build_weights(num_weights)
    ps = Particles(
        rng_key=None,
        env_fn=None,
        env=env,
        controller=controller,
        runner=runner,
        normalized_key=normalized_key,
        save_name="test_particles",
        weights=weights,
        save_dir=f"{data_dir()}/test",
    )
    key = random.PRNGKey(0)
    ps.update_key(key)
    return ps


all_particles = {}
for num_weights in [1, 5, 10, 20]:
    all_particles[num_weights] = build_particles(num_weights)


@pytest.mark.parametrize("num_weights", [1, 5])
def test_features(num_weights):
    ps = all_particles[num_weights]
    task = env.all_tasks[0]
    feats = ps.get_features(task, str(task))
    nfeats = len(env.features_keys)
    assert feats.shape == (nfeats, num_weights, T)


@pytest.mark.parametrize("num_weights", [1, 5])
def test_features_sum(num_weights):
    ps = all_particles[num_weights]
    task = env.all_tasks[0]
    feats_sum = ps.get_features_sum(task, str(task))
    nfeats = len(env.features_keys)
    assert feats_sum.shape == (nfeats, num_weights)


@pytest.mark.parametrize("num_weights", [1, 5])
def test_violations(num_weights):
    ps = all_particles[num_weights]
    task = env.all_tasks[0]
    vios_sum = ps.get_violations(task, str(task))
    nvios = len(env.constraints_keys)
    assert vios_sum.shape == (nvios, num_weights)


@pytest.mark.parametrize("num_weights", [1, 5])
def test_actions(num_weights):
    ps = all_particles[num_weights]
    task = env.all_tasks[0]
    udim = 2
    actions = ps.get_actions(task, str(task))
    assert actions.shape == (T, num_weights, udim)


@pytest.mark.parametrize("num_weights", [1, 5])
def test_dump_merge(num_weights):
    ps = all_particles[num_weights]
    task = env.all_tasks[0]
    udim = 2
    actions = ps.get_actions(task, str(task))
    data = ps.dump_task(task, str(task))
    new_ps = build_particles(num_weights)
    new_ps.merge(data)
    assert str(task) in new_ps.cached_names
    new_actions = new_ps.get_actions(task, str(task))
    assert onp.allclose(actions, new_actions)


@pytest.mark.parametrize("num_weights", [1, 5])
def test_save_load(num_weights):
    ps = all_particles[num_weights]
    task = env.all_tasks[0]
    udim = 2
    actions = ps.get_actions(task, str(task))
    ps.save()
    new_ps = build_particles(num_weights)
    new_ps.load()
    for key, val in ps.weights.items():
        assert onp.allclose(val, new_ps.weights[key])


@pytest.mark.parametrize("num_weights", [1, 5])
def test_compare_with(num_weights):
    ps = all_particles[num_weights]
    target = build_particles(1)
    task = env.all_tasks[0]
    diff_rews, diff_vios = ps.compare_with(task, str(task), target)
    assert diff_rews.shape == (num_weights,)
    assert diff_vios.shape == (num_weights,)


@pytest.mark.parametrize("num_weights", list(itertools.product([10, 20])))
def test_resample(num_weights):
    ps = all_particles[num_weights]
    resample = ps.resample(probs=onp.ones(num_weights))
    assert len(resample.weights) == num_weights


@pytest.mark.parametrize("num_weights", [1, 5])
def test_entropy(num_weights):
    ps = all_particles[num_weights]
    ent = ps.entropy(bins=5, max_weights=10)


@pytest.mark.parametrize(
    "num_weights, num_map", list(itertools.product([10, 20], [1, 2, 5]))
)
def test_map_estimate(num_weights, num_map):
    ps = all_particles[num_weights]
    map_ps = ps.map_estimate(num_map)
    assert len(map_ps.weights) == num_map
