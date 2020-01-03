"""Run Active-IRD Experiment Loop.

Note:
    * See (rdb.exps.active_ird.py) for more details.

"""
from rdb.exps.active_ird import ExperimentActiveIRD
from rdb.exps.acquire import ActiveInfoGain, ActiveRatioTest, ActiveRandom
from rdb.infer.ird_oc import IRDOptimalControl
from rdb.optim.mpc import shooting_method
from rdb.infer.utils import *
from functools import partial
from jax import random
import numpyro.distributions as dist
import gym, rdb.envs.drive2d

# RANDOM_KEYS = [1, 2, 3, 4]
# RANDOM_KEYS = [5, 6, 7, 8]
# RANDOM_KEYS = [9, 10, 11, 12]
RANDOM_KEYS = [13, 14, 15, 16]
# RANDOM_KEYS = [17, 18, 19, 20]
NUM_WARMUPS = 100
NUM_PROPOSAL_TASKS = 16

## Full scale sampling
NUM_NORMALIZERS = 200
# NUM_SAMPLES = 800
NUM_SAMPLES = 2000
NUM_ACTIVE_SAMPLES = 100
NUM_EVAL_SAMPLES = 100
# NUM_EVAL_TASKS = 8
NUM_EVAL_TASKS = 1

## Debug set True
USER_TRUE_W = True

## Faster sampling
# NUM_NORMALIZERS = 100
# NUM_SAMPLES = 200
# NUM_ACTIVE_SAMPLES = 25
# NUM_EVAL_SAMPLES = 50
# NUM_EVAL_TASKS = 8

## Testing
# NUM_NORMALIZERS = 3
# NUM_SAMPLES = 10
# NUM_ACTIVE_SAMPLES = 10
# NUM_EVAL_SAMPLES = 7
# NUM_EVAL_TASKS = 2


NUM_DESIGNERS = 20
NUM_ACTIVE_DESIGNERS = 4
MAX_WEIGHT = 8.0
BETA = 5.0
HORIZON = 10
EXP_ITERATIONS = 8
PROPOSAL_VAR = 0.2

env = gym.make("Week3_02-v0")
env.reset()
controller, runner = shooting_method(
    env, env.main_car.cost_runtime, HORIZON, env.dt, replan=False
)

# TODO: find true w
true_w = {
    "dist_cars": 1.0,
    "dist_lanes": 0.1,
    "dist_fences": 0.6,
    "speed": 0.5,
    "control": 0.16,
}
# Training Environment
task = (-0.4, 0.3)

""" Prior sampling & likelihood functions for PGM """
log_prior_dict = {
    "dist_cars": dist.Uniform(0.0, 0.01),
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
    true_w=true_w,
    prior_log_prob=prior_log_prob_fn,
    normalizer_fn=norm_sample_fn,
    proposal_fn=proposal_fn,
    sample_args={"num_warmups": NUM_WARMUPS, "num_samples": NUM_SAMPLES},
    designer_args={"num_warmups": NUM_WARMUPS, "num_samples": NUM_DESIGNERS},
    use_true_w=USER_TRUE_W,
)

""" Active acquisition function for experiment """
acquire_fns = {
    # "infogain": ActiveInfoGain(
    #     rng_key=None,
    #     env=env,
    #     model=ird_model,
    #     beta=BETA,
    #     num_active_designers=NUM_ACTIVE_DESIGNERS,
    #     num_active_sample=NUM_ACTIVE_SAMPLES,
    #     debug=False,
    # ),
    "ratiomean": ActiveRatioTest(
        rng_key=None,
        env=env,
        model=ird_model,
        method="mean",
        num_active_sample=NUM_ACTIVE_SAMPLES,
        debug=False,
    ),
    # "ratiomin": ActiveRatioTest(
    #     rng_key=None,
    #     env=env,
    #     model=ird_model,
    #     method="min",
    #     num_active_sample=NUM_ACTIVE_SAMPLES,
    #     debug=False,
    # ),
    # "random": ActiveRandom(rng_key=None, env=env, model=ird_model),
}

experiment = ExperimentActiveIRD(
    env,
    ird_model,
    acquire_fns,
    iterations=EXP_ITERATIONS,
    num_eval_tasks=NUM_EVAL_TASKS,
    num_eval_sample=NUM_EVAL_SAMPLES,
    num_proposal_tasks=NUM_PROPOSAL_TASKS,
    # Hard coded candidates
    # fixed_candidates=[(-0.4, -0.7), (-0.2, 0.5)],
    fixed_candidates=[(-0.2, 0.5)],
    # debug_belief_task=(-0.2, 0.5),
    # debug_belief_task=None,
    # save_dir="data/191221_true",
    save_dir="data/191231_test",
    exp_name="active_ird_exp1",
)

""" Experiment """
for ki in RANDOM_KEYS:
    key = random.PRNGKey(ki)
    experiment.update_key(key)
    experiment.run(task)
