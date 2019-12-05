import gym
import numpyro
import time, copy
import jax.numpy as np
import rdb.envs.drive2d
import numpyro.distributions as dist

from jax import random, vmap
from rdb.infer.ird_oc import *
from rdb.infer.algos import *
from rdb.infer.utils import *
from rdb.optim.mpc import shooting_optimizer
from rdb.optim.runner import Runner
from rdb.visualize.render import render_env
from rdb.visualize.preprocess import normalize_features

REPLAN = False


env = gym.make("Week3_02-v0")
env.reset()
cost_runtime = env.main_car.cost_runtime
horizon = 10
controller, runner = shooting_optimizer(
    env, cost_runtime, env.udim, horizon, env.dt, replan=REPLAN
)
beta = 5.0
env.reset()
y0_idx, y1_idx = 1, 5

# Training Environment
state_train = copy.deepcopy(env.state)
state_train[y0_idx] = -0.4
state_train[y1_idx] = 0.3

# Testing Environment 1
state_test1 = copy.deepcopy(env.state)
state_test1[y0_idx] = -0.4
state_test1[y1_idx] = -0.7

# Testing Environment 2
state_test2 = copy.deepcopy(env.state)
state_test2[y0_idx] = -0.2
state_test2[y1_idx] = 0.6


def prior_log_prob(state):
    w_log_dist_cars = 0.0
    log_dist_lanes = np.log(state["dist_lanes"])
    if log_dist_lanes < 0 or log_dist_lanes > 10:
        return -np.inf
    log_dist_fences = np.log(state["dist_fences"])
    if log_dist_fences < 0 or log_dist_fences > 10:
        return -np.inf
    log_speed = np.log(state["speed"])
    if log_speed < 0 or log_speed > 10:
        return -np.inf
    log_control = np.log(state["control"])
    if log_control < 0 or log_control > 10:
        return -np.inf
    return 0.0


def proposal(weight):
    std_dict = {"dist_lanes": 0.2, "dist_fences": 0.2, "speed": 0.2, "control": 0.2}
    next_weight = copy.deepcopy(weight)
    for key, val in next_weight.items():
        log_val = np.log(val)
        if key in std_dict.keys():
            std = std_dict[key]
            next_log_val = numpyro.sample("next_weight", dist.Normal(log_val, std))
            next_weight[key] = np.exp(next_log_val)
    return next_weight


user_weights = {
    "dist_cars": 100.0,
    "dist_lanes": 10.0,
    "dist_fences": 300.0,
    "speed": 20.0,
    "control": 80.0,
}


key = random.PRNGKey(1)
pgm = IRDOptimalControl(
    key, env, controller, runner, beta, prior_log_prob=prior_log_prob
)
sampler = MetropolisHasting(
    key, pgm, num_warmups=5, num_samples=20, proposal_fn=proposal
)
sampler.init(user_weights)


## Sample training environments
samples_train_dict = sampler.sample(user_weights, init_state=state_train)
samples_train = stack_dict_values(samples_train_dict)
# Remove 'dist_cars' (constant)
entropy = pgm.entropy(samples_train[:, 1:])


def infogain(samples_train_dict, state_test):
    # Sample testing environments 1
    h = 0.0
    for w_i, w_train in enumerate(samples_train_dict):
        print(f"Info {w_i + 1}/{len(samples_train_dict)}")
        sampler.init(w_train)
        samples_test = sampler.sample(w_train, init_state=state_test, verbose=True)
        h += pgm.entropy(stack_dict_values(samples_test)[:, 1:])
    h /= float(len(samples_train_dict) * len(samples_test))
    return -h


print("Sampling infogain: test 1")
gain1 = infogain(samples_train_dict, state_test1)
print(f"Test 1 gain {gain1:.3f} (should suck)")

print()
print("Sampling infogain: test 2")
gain2 = infogain(samples_train_dict, state_test2)
print(f"Test 2 gain {gain2:.3f} (should be better)")
