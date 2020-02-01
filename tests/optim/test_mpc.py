import gym
import copy
import pytest
import jax.numpy as np
import numpy as onp
import rdb.envs.drive2d
from rdb.optim.mpc import build_mpc
from rdb.optim.runner import Runner

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
    "dist_objects": 100.25,
    "speed": 0.05,
    "control": 0.1,
}
optimizer, runner = build_mpc(
    env, main_car.cost_runtime, horizon, env.dt, replan=False, T=T
)


def get_init_states(nbatch):
    num_tasks = len(env.all_tasks)
    tasks = env.all_tasks[onp.random.randint(0, num_tasks, size=nbatch)]
    states = onp.stack([env.get_init_state(t) for t in tasks], axis=0)
    return states


@pytest.mark.parametrize("nbatch", [1, 2, 5, 10])
def test_vectorized_mpc(nbatch):
    states = get_init_states(nbatch)
    ## Single
    costs_single = []
    for state_i in states:
        state_i = state_i[None, :]
        acs_i = optimizer(state_i, weights=weights, batch=False)
        traj_i, cost_i, info_i = runner(state_i, acs_i, weights=weights, batch=False)
        costs_single.append(cost_i)
    costs_single = onp.concatenate(costs_single)
    ## Batch
    weights_all = [weights] * nbatch
    acs_all = optimizer(states, weights=weights_all)
    traj_all, costs_all, info_all = runner(
        states, acs_all, weights=weights_all, batch=True
    )
    print(f"Diff costs for {nbatch}: {onp.sum(onp.abs(costs_all - costs_single)):.3f}")
    max_idx = onp.abs(costs_all - costs_single).argmax()
    max_ratio = onp.abs(costs_all - costs_single).max() / costs_single[max_idx]
    print(f"Diff costs max ratio: {max_ratio:.3f}")


@pytest.mark.parametrize("nbatch", [2, 1, 5, 10])
def test_scipy_mpc(nbatch):
    states = get_init_states(nbatch)
    weights_all = [weights] * nbatch
    acs_all = optimizer(states, weights=weights_all)
    traj_all, costs_all, info_all = runner(
        states, acs_all, weights=weights_all, batch=True
    )
    assert acs_all.shape == (horizon, nbatch, udim)
    assert traj_all.shape == (horizon, nbatch, xdim)
    assert info_all["costs"].shape == (nbatch, horizon)
    for key, val in info_all["feats"].items():
        assert val.shape == (nbatch, horizon)
    for key, val in info_all["feats_sum"].items():
        assert val.shape == (nbatch,)
    for key, val in info_all["violations"].items():
        assert val.shape == (nbatch, horizon)
    for key, val in info_all["vios_sum"].items():
        assert val.shape == (nbatch,)
    for key, val in info_all["metadata"].items():
        assert val.shape == (nbatch, horizon)
