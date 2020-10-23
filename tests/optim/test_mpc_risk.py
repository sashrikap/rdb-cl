import gym
import copy
import pytest
import jax.numpy as np
import numpy as onp
import rdb.envs.drive2d
from jax.config import config
from rdb.optim.mpc_risk import *
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
    "dist_objects": 100.25,
    "speed": 0.05,
    "control": 0.1,
}
controller, runner = build_risk_averse_mpc(
    env,
    main_car.cost_runtime,
    horizon,
    env.dt,
    replan=False,
    T=T,
    mode="trajwise",
    engine="scipy",
    method="lbfgs",
)
adam_controller, _ = build_risk_averse_mpc(
    env,
    main_car.cost_runtime,
    horizon,
    env.dt,
    replan=False,
    T=T,
    mode="stepwise",
    engine="jax",
    method="adam",
)
momt_controller, _ = build_risk_averse_mpc(
    env,
    main_car.cost_runtime,
    horizon,
    env.dt,
    replan=False,
    T=T,
    engine="jax",
    method="momentum",
)


def get_init_states(nbatch):
    num_tasks = len(env.all_tasks)
    tasks = env.all_tasks[onp.random.randint(0, num_tasks, size=nbatch)]
    states = onp.concatenate([env.get_init_state(t) for t in tasks], axis=0)
    return states


def run_single(states, controller, weights):
    costs_single = []
    for si, state_i in enumerate(states):
        state_i = state_i[None, :]
        acs_i = controller(state_i, batch=False, weights=weights[si])
        traj_i, cost_i, info_i = runner(
            state_i, acs_i, batch=False, weights=weights[si]
        )
        costs_single.append(cost_i)
    costs_single = onp.concatenate(costs_single)
    return costs_single


def run_batch(states, controller, weights):
    acs_all = controller(states, weights=weights)
    traj_all, costs_all, info_all = runner(states, acs_all, weights=weights)
    return costs_all


@pytest.mark.parametrize("nbatch", [2, 5, 1])
@pytest.mark.parametrize("nweights", [3, 4, 1])
def test_lbfgs_lbfgs(nbatch, nweights):
    risk_weights = [[weights] * nweights] * nbatch

    states = get_init_states(nbatch)
    ## Single
    costs_single = run_single(states, controller, risk_weights)
    ## Batch
    costs_batch = run_batch(states, controller, risk_weights)
    print(f"Nbatch {nbatch}")


test_lbfgs_lbfgs(2, 3)


@pytest.mark.parametrize("nbatch", [1, 2, 5])
@pytest.mark.parametrize("nweights", [1, 2, 4])
def test_adam_lbfgs(nbatch, nweights):
    risk_weights = [[weights] * nweights] * nbatch

    states = get_init_states(nbatch)
    ## Single
    costs_single = run_single(states, controller, risk_weights)
    ## Batch
    costs_batch = run_batch(states, adam_controller, risk_weights)
    print(f"Nbatch {nbatch}")


@pytest.mark.parametrize("nbatch", [1, 2, 5])
@pytest.mark.parametrize("nweights", [1, 2, 4])
def ttest_adam_adam(nbatch, nweights):
    risk_weights = [[weights] * nweights] * nbatch

    states = get_init_states(nbatch)
    ## Single
    costs_single = run_single(states, adam_controller, risk_weights)
    ## Batch
    costs_batch = run_batch(states, adam_controller, risk_weights)
    print(f"Nbatch {nbatch}")


@pytest.mark.parametrize("nbatch", [1, 2, 5])
@pytest.mark.parametrize("nweights", [1, 2, 4])
def test_momentum_lbfgs(nbatch, nweights):
    risk_weights = [[weights] * nweights] * nbatch

    states = get_init_states(nbatch)
    ## Single
    costs_single = run_single(states, controller, risk_weights)
    ## Batch
    costs_batch = run_batch(states, momt_controller, risk_weights)
    print(f"Nbatch {nbatch}")


@pytest.mark.parametrize("nbatch", [1, 2, 5])
@pytest.mark.parametrize("nweights", [1, 2, 4])
def test_scipy_mpc(nbatch, nweights):
    risk_weights = [[weights] * nweights] * nbatch

    states = get_init_states(nbatch)
    acs_all = controller(states, weights=risk_weights)
    traj_all, costs_all, info_all = runner(states, acs_all, weights=risk_weights)
    assert acs_all.shape == (nbatch, horizon, udim)
    assert traj_all.shape == (nbatch, horizon, xdim)
    assert info_all["costs"].shape == (nbatch, nweights, horizon)
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
