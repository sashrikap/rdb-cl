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
from rdb.optim.mpc_risk import build_risk_averse_mpc
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
            if key != "bias":
                if key == normalized_key:
                    w[key] = 1.0
                else:
                    w[key] = onp.random.random()
        weights.append(w)
    return DictList(weights)


def build_particles(num_weights, risk=False):
    env = gym.make("Week6_02-v1")  # Two Blockway
    env.reset()
    main_car = env.main_car
    if risk:
        controller, runner = build_risk_averse_mpc(
            env,
            main_car.cost_runtime,
            T,
            env.dt,
            replan=False,
            T=10,
            engine="jax",
            method="adam",
            test_mode=True,
            # test_mode=False,
        )
    else:
        controller, runner = build_mpc(
            env,
            main_car.cost_runtime,
            T,
            env.dt,
            replan=False,
            T=10,
            engine="jax",
            method="adam",
            test_mode=True,
            # test_mode=False,
        )
    weights = build_weights(num_weights)
    ps = Particles(
        rng_name="",
        rng_key=None,
        env_fn=None,
        env=env,
        controller=controller,
        runner=runner,
        normalized_key=normalized_key,
        save_name="test_particles",
        weights=weights,
        save_dir=f"{data_dir()}/test",
        weight_params={"bins": 10, "max_weights": 20},
        risk_averse=risk,
    )
    key = random.PRNGKey(0)
    ps.update_key(key)
    return ps


all_particles = {}
rsk_particles = {}
for num_weights in [1, 5, 10, 20]:
    all_particles[num_weights] = build_particles(num_weights)
    rsk_particles[num_weights] = build_particles(num_weights, risk=True)


@pytest.mark.parametrize("num_weights", [10, 20])
def test_digitize(num_weights):
    ps = build_particles(num_weights)
    which_bins = ps.digitize(bins=5, max_weights=1, log_scale=True, matrix=True)


@pytest.mark.parametrize("num_weights", [2, 5])
def test_index(num_weights):
    ps = build_particles(num_weights)
    tasks = [env.all_tasks[0]]
    feats = ps.get_features(tasks)
    nfeats = len(env.features_keys)
    idx = 0
    # import pdb; pdb.set_trace()
    ps1 = ps[idx]
    assert len(ps1.weights) == 1
    idx = [0, 1]
    ps2 = ps[idx]
    assert len(ps2.weights) == 2


@pytest.mark.parametrize("num_weights", [1, 5])
def test_features(num_weights):
    ps = all_particles[num_weights]
    tasks = [env.all_tasks[0]]
    feats = ps.get_features(tasks)
    nfeats = len(env.features_keys)
    assert feats.shape == (1, num_weights, T)
    assert feats.num_keys == nfeats
    feats = ps.get_features(tasks, lower=True)
    assert feats.shape == (1, num_weights, T)
    assert feats.num_keys == nfeats


@pytest.mark.parametrize("num_weights", [1, 5])
def test_features_sum(num_weights):
    ps = all_particles[num_weights]
    tasks = [env.all_tasks[0]]
    nfeats = len(env.features_keys)
    feats_sum = ps.get_features_sum(tasks)
    assert feats_sum.shape == (1, num_weights)
    assert feats_sum.num_keys == nfeats
    feats_sum = ps.get_features_sum(tasks, lower=True)
    assert feats_sum.shape == (1, num_weights)
    assert feats_sum.num_keys == nfeats


@pytest.mark.parametrize("num_weights", [1, 5])
def test_violations(num_weights):
    ps = all_particles[num_weights]
    tasks = [env.all_tasks[0]]
    nvios = len(env.constraints_keys)
    vios_sum = ps.get_violations(tasks)
    assert vios_sum.shape == (1, num_weights)
    assert vios_sum.num_keys == nvios
    vios_sum = ps.get_violations(tasks, lower=True)
    assert vios_sum.shape == (1, num_weights)
    assert vios_sum.num_keys == nvios


@pytest.mark.parametrize("num_weights", [1, 5])
def test_actions(num_weights):
    ps = all_particles[num_weights]
    tasks = [env.all_tasks[0]]
    udim = 2
    actions = ps.get_actions(tasks)
    assert actions.shape == (1, num_weights, T, udim)


@pytest.mark.parametrize("num_weights", [1, 5])
def test_costs(num_weights):
    ps = all_particles[num_weights]
    tasks = [env.all_tasks[0]]
    nfeats = len(env.features_keys)
    costs = ps.get_costs(tasks)
    assert costs.shape == (1, num_weights)
    costs = ps.get_costs(tasks, lower=True)
    assert costs.shape == (1, num_weights)


@pytest.mark.parametrize("num_weights", [1, 2, 4, 5])
@pytest.mark.parametrize("num_tasks", [1, 3, 7])
def test_risk(num_weights, num_tasks):
    ps_obs = rsk_particles[1]
    tasks = env.all_tasks[:num_tasks]
    feats_keys = env.features_keys
    feats_costs = ps_obs.get_costs(tasks)


@pytest.mark.parametrize("num_weights", [1, 5])
def test_offset(num_weights):
    ps_obs = all_particles[1]
    tasks = env.all_tasks[:1]
    feats_keys = env.features_keys
    feats_dict = ps_obs.get_features(tasks).prepare(feats_keys)
    feats_sum = feats_dict.sum(axis=-1)

    ps = all_particles[num_weights]
    ps_sub = ps.subsample(num_weights)
    offsets = ps_sub.get_offset_by_features(feats_dict)
    assert offsets.shape == (num_weights,)
    ps_sub.weights = ps_sub.weights.add_key("bias", offsets)
    ## Assert offset truly zero
    costs = ps_sub.get_costs(tasks)
    assert costs.shape == (1, num_weights)


@pytest.mark.parametrize("num_weights", [1, 5])
@pytest.mark.parametrize("num_tile", [1, 2, 3])
@pytest.mark.parametrize("num_tasks", [4, 6])
def test_tile(num_weights, num_tile, num_tasks):
    ps = all_particles[num_weights]
    tasks = env.all_tasks[:num_tasks]
    feats = ps.get_features(tasks)
    new_ps = ps.tile(num_tile)
    assert len(new_ps.cached_names) == len(ps.cached_names)
    new_feats = new_ps.get_features(tasks)
    old_feats = ps.get_features(tasks)

    assert old_feats.shape[:2] == (num_tasks, num_weights)
    assert new_feats.shape[1] == num_tile * old_feats.shape[1]

    new_feats_sum = new_ps.get_features_sum(tasks)
    old_feats_sum = ps.get_features_sum(tasks)

    assert old_feats_sum.shape[:2] == (num_tasks, num_weights)
    assert new_feats_sum.shape[1] == num_tile * old_feats_sum.shape[1]

    new_actions = new_ps.get_actions(tasks)
    old_actions = ps.get_actions(tasks)
    assert new_actions.shape[1] == num_tile * old_actions.shape[1]
    new_vios = new_ps.get_violations(tasks)
    old_vios = ps.get_violations(tasks)
    assert new_vios.shape[1] == num_tile * old_vios.shape[1]


@pytest.mark.parametrize("num_weights", [1, 5])
def test_combine(num_weights):
    ps = build_particles(num_weights)
    tasks0 = [env.all_tasks[0]]
    tasks1 = [env.all_tasks[1]]
    tasks2 = [env.all_tasks[2]]
    udim = 2
    ps.get_actions(tasks0)
    ps.get_actions(tasks1)
    new_ps = build_particles(1)
    new_ps.get_actions(tasks1)
    new_ps.get_actions(tasks2)
    com_ps = ps.combine(new_ps)
    assert len(ps.cached_names) == 2
    assert len(new_ps.cached_names) == 2
    assert len(com_ps.cached_names) == 1


@pytest.mark.parametrize("num_weights", [1, 5])
@pytest.mark.parametrize("num_repeat", [2, 5])
def test_repeat(num_weights, num_repeat):
    ps = build_particles(num_weights)
    tasks0 = [env.all_tasks[0]]
    udim = 2
    ps.get_actions(tasks0)
    n_repeat = 5
    new_ps = ps.repeat(num_repeat)
    assert len(new_ps.weights) == num_weights * num_repeat


@pytest.mark.parametrize("num_weights", [1, 5])
def test_dump_merge(num_weights):
    ps = all_particles[num_weights]
    tasks = [env.all_tasks[0]]
    udim = 2
    actions = ps.get_actions(tasks)
    new_ps = build_particles(num_weights)
    data = ps.dump_tasks(tasks)
    new_ps.merge_tasks(tasks, data)
    for task in tasks:
        assert new_ps.get_task_name(task) in new_ps.cached_names
    new_actions = new_ps.get_actions(tasks)
    assert onp.allclose(actions, new_actions)


@pytest.mark.parametrize("num_weights", [1, 5])
def test_save_load(num_weights):
    ps = all_particles[num_weights]
    tasks = [env.all_tasks[0]]
    udim = 2
    actions = ps.get_actions(tasks)
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
    ## Compare with target
    comparisons = ps.compare_with(task, target)
    assert comparisons["rews"].shape == (num_weights,)
    assert comparisons["vios"].shape == (num_weights,)
    assert comparisons["vios_by_name"].shape == (num_weights,)
    assert comparisons["vios_by_name"].num_keys > 0

    ## Compare w/o target
    notarget_comparisons = ps.compare_with(task, target=None)
    assert notarget_comparisons["rews"].shape == (num_weights,)
    assert notarget_comparisons["vios"].shape == (num_weights,)
    assert notarget_comparisons["vios_by_name"].shape == (num_weights,)
    assert notarget_comparisons["vios_by_name"].num_keys > 0


@pytest.mark.parametrize("num_weights", [10, 20])
def test_resample(num_weights):
    ps = all_particles[num_weights]
    resample = ps.resample(probs=onp.ones(num_weights))
    assert len(resample.weights) == num_weights


@pytest.mark.parametrize("num_weights", [1, 5])
def test_entropy(num_weights):
    ps = all_particles[num_weights]
    ent = ps.entropy(bins=5, max_weights=10, log_scale=True)


@pytest.mark.parametrize("num_weights", [10, 20])
@pytest.mark.parametrize("num_map", [1, 2, 5])
def test_map_estimate(num_weights, num_map):
    ps = all_particles[num_weights]
    map_ps = ps.map_estimate(num_map)
    assert len(map_ps.weights) == num_map
