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
import gym, rdb.envs.drive2d
import numpyro.distributions as dist

RANDOM_KEYS = [1, 2, 3, 4, 5, 6]
NUM_SAMPLES = 1000
NUM_NORMALIZERS = 200
MAX_WEIGHT = 5.0
BETA = 5.0
HORIZON = 10
EXP_ITERATIONS = 1
PROPOSAL_VAR = 0.01

env = gym.make("Week3_02-v0")
env.reset()
controller, runner = shooting_method(
    env, env.main_car.cost_runtime, env.udim, HORIZON, env.dt, replan=False
)

# TODO: find true w
true_w = {}

""" Prior sampling & likelihood functions for PGM """
log_prior_dict = {
    "dist_cars": dist.Uniform(0.0, 0.01),
    "dist_lanes": dist.Uniform(-MAX_WEIGHT, MAX_WEIGHT),
    "dist_fences": dist.Uniform(-MAX_WEIGHT, MAX_WEIGHT),
    "speed": dist.Uniform(-MAX_WEIGHT, MAX_WEIGHT),
    "control": dist.Uniform(-MAX_WEIGHT, MAX_WEIGHT),
}
proposal_std_dict = {
    "dist_lanes": PROPOSAL_VAR,
    "dist_fences": PROPOSAL_VAR,
    "speed": PROPOSAL_VAR,
    "control": PROPOSAL_VAR,
}
prior_log_prob_fn = partial(prior_log_prob, prior_dict=log_prior_dict)
prior_sample_fn = partial(prior_sample, prior_dict=log_prior_dict)
norm_sample_fn = partial(
    normalizer_sample, sample_fn=prior_sample_fn, num=NUM_NORMALIZERS
)
proposal_fn = partial(gaussian_proposal, std_dict=proposal_std_dict)
ird_model = IRDOptimalControl(
    rng_key=None,
    env=env,
    controller=controller,
    runner=runner,
    beta=BETA,
    prior_log_prob=prior_log_prob_fn,
    normalizer_fn=norm_sample_fn,
    sample_args={
        "num_warmups": 100,
        "num_samples": NUM_SAMPLES,
        "proposal_fn": proposal_fn,
    },
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
