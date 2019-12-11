import gym
import numpyro
import time, copy
import jax.numpy as np
import numpy as onp
import rdb.envs.drive2d
import numpyro.distributions as dist
import matplotlib.pyplot as plt

from functools import partial
from tqdm.auto import tqdm
from jax import random, vmap
from rdb.infer.ird_oc import *
from rdb.infer.algos import *
from rdb.infer.utils import *
from rdb.optim.mpc import shooting_method
from rdb.optim.runner import Runner
from rdb.visualize.render import render_env
from numpyro.handlers import scale, condition, seed
from rdb.visualize.preprocess import normalize_features

REPLAN = False
RANDOM_KEYS = [1, 2, 3, 4, 5, 6]
NUM_SAMPLES = 1000
PLOT_BINS = 100
NUM_DESIGNER_SAMPLES = 20
NUM_NORMALIZERS = 100
MAX_WEIGHT = 5.0
BETA = 5.0

env = gym.make("Week3_02-v0")
env.reset()
cost_runtime = env.main_car.cost_runtime
horizon = 10
controller, runner = shooting_method(env, cost_runtime, horizon, env.dt, replan=REPLAN)
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


log_prior_dict = {
    "dist_cars": dist.Uniform(0.0, 0.01),
    "dist_lanes": dist.Uniform(-MAX_WEIGHT, MAX_WEIGHT),
    "dist_fences": dist.Uniform(-MAX_WEIGHT, MAX_WEIGHT),
    "speed": dist.Uniform(-MAX_WEIGHT, MAX_WEIGHT),
    "control": dist.Uniform(-MAX_WEIGHT, MAX_WEIGHT),
}

""" Prior sampling & likelihood functions for PGM """
prior_log_prob_fn = partial(prior_log_prob, prior_dict=log_prior_dict)
prior_sample_fn = partial(prior_sample, prior_dict=log_prior_dict)
norm_sample_fn = partial(
    normalizer_sample, sample_fn=prior_sample_fn, num=NUM_NORMALIZERS
)


def proposal(weight, const=0.01):
    std_dict = {
        "dist_lanes": const,
        "dist_fences": const,
        "speed": const,
        "control": const,
    }
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


def infogain(
    samples_train_dict,
    state_test,
    state_name,
    num_samples=NUM_DESIGNER_SAMPLES,
    verbose=False,
):
    # Sample testing environments 1
    neg_ents = []
    pgm.initialize(state_test, state_name)
    range_ = tqdm(samples_train_dict, desc="InfoGain envs")
    for w_i, w_train in enumerate(range_):
        sampler.init(w_train)
        samples_test = sampler.sample(
            w_train,
            init_state=state_test,
            verbose=verbose,
            state_name=state_name,
            num_samples=num_samples,
        )
        neg_ents.append(-1 * pgm.entropy(stack_dict_values_ratio(samples_test)[:, 1:]))
    return onp.array(neg_ents)


def plot_samples(samples_dicts, highlight_dict=None):
    plt.figure()
    n_values = len(samples_dicts[0].values())
    for i, key in enumerate(samples_dicts[0].keys()):
        values = [s[key] for s in samples_dicts]
        plt.subplot(n_values, 1, i + 1)
        n, bins, patches = plt.hist(
            values, PLOT_BINS, density=True, facecolor="b", alpha=0.75
        )
        ## Highlight value
        if highlight_dict is not None:
            val = highlight_dict[key]
            bin_i = np.argmin(np.abs(bins[:-1] - val))
            patches[bin_i].set_fc("r")
        plt.title(key)
    plt.tight_layout()
    plt.show()


""" Plot 1 example """
key = random.PRNGKey(1)
pgm = IRDOptimalControl(
    key, env, controller, runner, BETA, prior_log_prob=prior_log_prob
)
proposal_fn = partial(proposal, const=0.01)
sampler = MetropolisHasting(
    key, pgm, num_warmups=100, num_samples=NUM_SAMPLES, proposal_fn=proposal_fn
)
sampler.init(user_weights)
norm_sample_fn = seed(norm_sample_fn, key)
pgm.initialize(state_train, "train", norm_sample_fn())
## Sample training environments
samples_train_dict = sampler.sample(
    user_weights, init_state=state_train, state_name="train", num_samples=NUM_SAMPLES
)
plot_samples(samples_train_dict, user_weights)


""" Demo 1 """
all_gain1 = []
all_gain2 = []
DEMO1_NUM_SAMPLES = 50
DEMO1_NUM_DESIGNER_SAMPLES = 40
DEMO1_PLOT = True
for ki, keyi in enumerate(RANDOM_KEYS):
    ## Create new key
    key = random.PRNGKey(keyi)

    ## Create PGM & Sampler
    pgm = IRDOptimalControl(
        key,
        env,
        controller,
        runner,
        BETA,
        prior_log_prob=prior_log_prob,
        normalizer_fn=norm_sample_fn,
    )
    pgm.initialize(state_train, "train")
    proposal_fn = partial(proposal, const=0.01)
    sampler = MetropolisHasting(
        key,
        pgm,
        num_warmups=100,
        num_samples=DEMO1_NUM_SAMPLES,
        proposal_fn=proposal_fn,
    )
    sampler.init(user_weights)

    ## Sample training environments
    samples_train_dict = sampler.sample(
        user_weights, state_name="train", init_state=state_train
    )
    if DEMO1_PLOT:
        plot_samples(samples_train_dict, user_weights)
    samples_train = stack_dict_values_ratio(samples_train_dict, user_weights)
    ## Remove 'dist_cars' (constant) from entropy calculation
    entropy = pgm.entropy(samples_train[:, 1:])

    print("Sampling infogain: test 1")
    gain1 = infogain(
        samples_train_dict, state_test1, "test1", DEMO1_NUM_DESIGNER_SAMPLES, False
    )
    all_gain1.append(gain1)
    print(f"Test 1 gain {gain1:.3f} (should suck)")
    print("Sampling infogain: test 2")
    gain2 = infogain(
        samples_train_dict, state_test2, "test2", DEMO1_NUM_DESIGNER_SAMPLES, False
    )
    all_gain2.append(gain2)
    print(f"Test 2 gain {gain2:.3f} (should be better)")
    print()

    print(f"Key {ki}/{len(RANDOM_KEYS)}")
    print(
        f"Gain 1 mean {np.array(all_gain1).mean():.3f} std {np.array(all_gain1).std():.3f}"
    )
    print(
        f"Gain 2 mean {np.array(all_gain2).mean():.3f} std {np.array(all_gain2).std():.3f}"
    )

    print()
