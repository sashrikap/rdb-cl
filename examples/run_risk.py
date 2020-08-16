import gym
import jax
import copy
import jax.numpy as np
import numpy as onp
import rdb.envs.drive2d

from time import time, sleep
from tqdm import tqdm
from rdb.optim.mpc_risk import build_risk_averse_mpc
from rdb.optim.mpc import build_mpc
from rdb.optim.runner import Runner
from rdb.infer import *
from rdb.visualize.render import render_env
from rdb.visualize.preprocess import normalize_features

REPLAN = -1
# ENGINE = "scipy"
# METHOD = "lbfgs"
ENGINE = "jax"
METHOD = "adam"
ENV_NAME = "Week6_02-v1"  # Two Blockway
TASK = (-0.7, -0.7, 0.13, 0.4, -0.13, 0.4)

env = gym.make(ENV_NAME)
env.reset()
main_car = env.main_car
horizon = 10
T = 10
weights1 = {
    "control": 0.1,
    "dist_cars": 0.1,
    "dist_fences": 2.675,
    "dist_lanes": 1.25,
    "dist_objects": 1.125,
    "speed": 2.5,
    # "speed_over": 40.0,
}
weights2 = {
    "control": 0.1,
    "dist_cars": 10.0,
    "dist_fences": 2.675,
    "dist_lanes": 1.25,
    "dist_objects": 1.125,
    "speed": 2.5,
    # "speed_over": 40.0,
}
obs_weights = weights2
true_weights = {
    "dist_cars": 1.0,
    "dist_lanes": 1.25,
    "dist_fences": 2.675,
    "dist_objects": 1.125,
    "speed": 2.5,
    "control": 2.0,
    # "speed_over": 40.0,
}


env.set_task(TASK)
env.reset()

print(f"Task {TASK}")

optimizer_risk_step, runner = build_risk_averse_mpc(
    env,
    main_car.cost_runtime,
    horizon,
    env.dt,
    replan=REPLAN,
    T=T,
    engine=ENGINE,
    method=METHOD,
)
optimizer_risk_traj, _ = build_risk_averse_mpc(
    env,
    main_car.cost_runtime,
    horizon,
    env.dt,
    replan=REPLAN,
    T=T,
    engine=ENGINE,
    method=METHOD,
    cost_args={"mode": "trajwise"},
)
optimizer, _ = build_mpc(
    env,
    main_car.cost_runtime,
    horizon,
    env.dt,
    replan=REPLAN,
    T=T,
    engine=ENGINE,
    method=METHOD,
)
state = copy.deepcopy(env.state)
test_state = env.get_init_state([0.5, 0.5, -0.12, 0.6, 0.8, 1.1])

w_list = DictList([weights1, weights2, true_weights])
actions_true = optimizer(test_state, weights=true_weights, batch=False)
_, cost_true, info_true = runner(
    test_state, actions_true, weights=true_weights, batch=False
)


actions_risk_step = optimizer_risk_step(test_state, weights=w_list)
_, cost_risk, info_risk = runner(
    test_state, actions_risk_step, weights=true_weights, batch=False
)
print("Cost risk (no offset)", cost_risk - cost_true)


## Populate offset and compare two types of planning

offset = [None, None, None]
offset[0] = (
    -1
    * (
        DictList([weights1]).prepare(env.features_keys).numpy_array()
        * info_true["feats_sum"].numpy_array()
    ).sum()
)
offset[1] = (
    -1
    * (
        DictList([weights2]).prepare(env.features_keys).numpy_array()
        * info_true["feats_sum"].numpy_array()
    ).sum()
)
offset[2] = (
    -1
    * (
        DictList([true_weights]).prepare(env.features_keys).numpy_array()
        * info_true["feats_sum"].numpy_array()
    ).sum()
)
w_list = w_list.add_key("bias", offset)
actions_risk_step = optimizer_risk_step(test_state, weights=w_list)
_, cost_risk_step, _ = runner(
    test_state, actions_risk_step, weights=true_weights, batch=False
)
print("Cost risk step (with offset)", cost_risk_step - cost_true)


actions_risk_traj = optimizer_risk_traj(test_state, weights=w_list)
_, cost_risk_traj, _ = runner(
    test_state, actions_risk_traj, weights=true_weights, batch=False
)
print("Cost risk traj (with offset)", cost_risk_traj - cost_true)


actions1 = optimizer(test_state, weights=weights1, batch=False)
_, cost1, _ = runner(test_state, actions1, weights=true_weights, batch=False)
print("Cost 1", cost1 - cost_true)
actions2 = optimizer(test_state, weights=weights2, batch=False)
_, cost2, _ = runner(test_state, actions2, weights=true_weights, batch=False)
print("Cost 2", cost2 - cost_true)
