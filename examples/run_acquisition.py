"""Run Active-IRD Experiment Loop.

Note:
    * See (rdb.exps.active_ird.py) for more details.

"""
from rdb.exps.active_ird import ExperimentActiveIRD
from rdb.exps.acquire import ActiveInfoGain, ActiveRatioTest
from rdb.optim.mpc import shooting_method
from rdb.infer.ird_oc import IRDOptimalControl
from rdb.infer.utils import *
from functools import partial
from jax import random
import gym, rdb.envs.drive2d
import numpyro.distributions as dist

RANDOM_KEYS = [1, 2, 3, 4, 5, 6]
NUM_SAMPLES = 1000
NUM_NORMALIZERS = 20
MAX_WEIGHT = 8.0
BETA = 5.0
HORIZON = 10
EXP_ITERATIONS = 1
PROPOSAL_VAR = 0.01

env = gym.make("Week3_02-v0")
env.reset()
controller, runner = shooting_method(
    env, env.main_car.cost_runtime, HORIZON, env.dt, replan=False
)

# TODO: find true w
true_w = {
    "dist_cars": 1.0,
    "dist_lanes": 0.1,
    "dist_fences": 3.0,
    "speed": 0.2,
    "control": 0.8,
}
# Training Environment
task = (-0.4, 0.3)

""" Prior sampling & likelihood functions for PGM """
log_prior_dict = {
    "dist_cars": dist.Uniform(0.0, 0.05),
    "dist_lanes": dist.Uniform(-MAX_WEIGHT, MAX_WEIGHT),
    "dist_fences": dist.Uniform(-MAX_WEIGHT, MAX_WEIGHT),
    "speed": dist.Uniform(-MAX_WEIGHT, MAX_WEIGHT),
    "control": dist.Uniform(-MAX_WEIGHT, MAX_WEIGHT),
}
proposal_std_dict = {
    "dist_cars": 1e-6,
    "dist_lanes": PROPOSAL_VAR,
    "dist_fences": PROPOSAL_VAR,
    "speed": PROPOSAL_VAR,
    "control": PROPOSAL_VAR,
}
prior_log_prob_fn = partial(prior_log_prob, log_prior_dict=log_prior_dict)
prior_sample_fn = partial(prior_sample, log_prior_dict=log_prior_dict)
norm_sample_fn = partial(
    normalizer_sample, sample_fn=prior_sample_fn, num=NUM_NORMALIZERS
)
proposal_fn = partial(gaussian_proposal, log_std_dict=proposal_std_dict)
ird_model = IRDOptimalControl(
    rng_key=None,
    env=env,
    controller=controller,
    runner=runner,
    beta=BETA,
    prior_log_prob=prior_log_prob_fn,
    normalizer_fn=norm_sample_fn,
    proposal_fn=proposal_fn,
    sample_args={"num_warmups": 100, "num_samples": NUM_SAMPLES},
    designer_args={"num_warmups": 5, "num_samples": 5},
)

""" Active acquisition function for experiment """
acquire_fns = {
    "infogain": ActiveInfoGain(env, ird_model),
    "ratiomax": ActiveRatioTest(env, ird_model, method="max"),
    # "ratiomin": ActiveRatioTest(env, method="min"),
}

experiment = ExperimentActiveIRD(
    env,
    true_w,
    ird_model,
    acquire_fns,
    eval_tasks=env.all_tasks,
    iterations=EXP_ITERATIONS,
)

""" Experiment """
for ki in RANDOM_KEYS:
    key = random.PRNGKey(ki)
    experiment.update_key(key)
    experiment.run(task)
