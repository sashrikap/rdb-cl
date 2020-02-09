from rdb.envs.drive2d.core.dynamics import *
from rdb.envs.drive2d.core import feature, constraints
from rdb.optim.utils import *
from rdb.infer import *
import itertools
import numpy as onp
import pytest
import gym


features_keys = ["dist_cars", "dist_lanes", "speed", "control", "control_thrust"]

env = gym.make("Week6_02-v1")


def build_constraints():
    constraints_dict = OrderedDict()
    constraints_dict["offtrack"] = constraints.build_offtrack(env=env)
    constraints_dict["overspeed"] = constraints.build_overspeed(env=env, max_speed=1.0)
    constraints_dict["underspeed"] = constraints.build_underspeed(
        env=env, min_speed=-0.2
    )
    constraints_dict["uncomfortable"] = constraints.build_uncomfortable(
        env=env, max_actions=env._control_bound
    )
    constraints_dict["wronglane"] = constraints.build_wronglane(env=env, lane_idx=2)
    constraints_dict["collision"] = constraints.build_collision(env=env)
    constraints_dict["overtake"] = constraints.build_overtake(env=env, car_idx=1)
    constraints_fn = merge_dict_funcs(constraints_dict)
    return constraints_fn


@pytest.mark.parametrize("batch", [2, 3, 4, 10])
def test_combined_full(batch):
    cons_fn = build_constraints()
    dyn_fn = env.dynamics_fn
    state = env.state.repeat(batch, axis=0)
    horizon = 20
    all_states, all_acs = [], []
    for i in range(horizon):
        ac = onp.zeros((batch, 2))
        state = dyn_fn(state, ac)
        all_states.append(state)
        all_acs.append(ac)
    all_states = onp.array(all_states)
    all_acs = onp.array(all_acs)
    cons_out = cons_fn(all_states, all_acs)
    for key, val in cons_out.items():
        assert val.shape == (horizon, batch)
