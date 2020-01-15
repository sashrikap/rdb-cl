"""Run Active-IRD Experiment Loop.

Note:
    * See (rdb.exps.active_ird.py) for more details.

"""
from rdb.exps.active_ird import ExperimentActiveIRD
from rdb.exps.acquire import ActiveInfoGain, ActiveRatioTest, ActiveRandom
from rdb.infer.ird_oc import IRDOptimalControl
from rdb.optim.mpc import shooting_method
from rdb.distrib.particles import ParticleServer
from rdb.infer.utils import *
from rdb.exps.utils import load_params
from functools import partial
from jax import random
import numpyro.distributions as dist
import yaml, argparse


def main():
    def env_fn():
        import gym, rdb.envs.drive2d

        env = gym.make(ENV_NAME)
        env.reset()
        return env

    def controller_fn(env):
        controller, runner = shooting_method(
            env, env.main_car.cost_runtime, HORIZON, env.dt, replan=False
        )
        return controller, runner

    # # TODO: find true w
    # true_w = {
    #     "dist_cars": 1.0,
    #     "dist_lanes": 0.1,
    #     "dist_fences": 0.6,
    #     "speed": 0.5,
    #     "control": 0.16,
    # }
    # # Training Environment
    # task = (-0.4, 0.3)

    # TODO: find true w
    # true_w = {
    #     "dist_cars": 1.0,
    #     "dist_lanes": 0.1,
    #     "dist_fences": 0.6,
    #     "speed": 0.5,
    #     "control": 0.16,
    # }
    # # Training Environment
    # task = (-0.4, 0.3)

    """ Prior sampling & likelihood functions for PGM """
    # log_prior_dict = {
    #     "dist_cars": dist.Uniform(0.0, 0.01),
    #     "dist_lanes": dist.Uniform(-MAX_WEIGHT, MAX_WEIGHT),
    #     "dist_fences": dist.Uniform(-MAX_WEIGHT, MAX_WEIGHT),
    #     "speed": dist.Uniform(-MAX_WEIGHT, MAX_WEIGHT),
    #     "control": dist.Uniform(-MAX_WEIGHT, MAX_WEIGHT),
    # }
    log_prior_dict = {
        "dist_cars": dist.Uniform(0.0, 0.01),
        "dist_lanes": dist.Uniform(-MAX_WEIGHT, MAX_WEIGHT),
        "dist_fences": dist.Uniform(-MAX_WEIGHT, MAX_WEIGHT),
        "dist_objects": dist.Uniform(-MAX_WEIGHT, MAX_WEIGHT),
        "speed": dist.Uniform(-MAX_WEIGHT, MAX_WEIGHT),
        "control": dist.Uniform(-MAX_WEIGHT, MAX_WEIGHT),
    }
    proposal_std_dict = {
        "dist_cars": 1e-6,
        "dist_lanes": PROPOSAL_VAR,
        "dist_fences": PROPOSAL_VAR,
        "dist_objects": PROPOSAL_VAR,
        "speed": PROPOSAL_VAR,
        "control": PROPOSAL_VAR,
    }
    prior_log_prob_fn = partial(prior_log_prob, log_prior_dict=log_prior_dict)
    prior_sample_fn = partial(prior_sample, log_prior_dict=log_prior_dict)
    norm_sample_fn = partial(
        normalizer_sample, sample_fn=prior_sample_fn, num=NUM_NORMALIZERS
    )
    proposal_fn = partial(gaussian_proposal, log_std_dict=proposal_std_dict)
    eval_server = ParticleServer(
        env_fn, controller_fn, num_workers=NUM_EVAL_WORKERS, parallel=PARALLEL
    )

    ird_model = IRDOptimalControl(
        rng_key=None,
        env_fn=env_fn,
        controller_fn=controller_fn,
        eval_server=eval_server,
        beta=BETA,
        true_w=TRUE_W,
        prior_log_prob_fn=prior_log_prob_fn,
        normalizer_fn=norm_sample_fn,
        proposal_fn=proposal_fn,
        sample_args={"num_warmups": NUM_WARMUPS, "num_samples": NUM_SAMPLES},
        designer_args={"num_warmups": NUM_WARMUPS, "num_samples": NUM_DESIGNERS},
        use_true_w=USER_TRUE_W,
        test_mode=TEST_MODE,
    )

    """ Active acquisition function for experiment """
    acquire_fns = {
        "infogain": ActiveInfoGain(
            rng_key=None, model=ird_model, beta=BETA, debug=False
        ),
        "ratiomean": ActiveRatioTest(
            rng_key=None, model=ird_model, method="mean", debug=False
        ),
        "ratiomin": ActiveRatioTest(
            rng_key=None, model=ird_model, method="min", debug=False
        ),
        "random": ActiveRandom(rng_key=None, model=ird_model),
    }

    SAVE_ROOT = "data" if not GCP_MODE else "/gcp_output"  # Don'tchange this line
    DEBUG_ROOT = "data" if not GCP_MODE else "/gcp_input"
    experiment = ExperimentActiveIRD(
        ird_model,
        acquire_fns,
        eval_server=eval_server,
        iterations=EXP_ITERATIONS,
        num_eval_tasks=NUM_EVAL_TASKS,
        num_eval_sample=NUM_EVAL_SAMPLES,
        num_active_tasks=NUM_ACTIVE_TASKS,
        num_active_sample=NUM_ACTIVE_SAMPLES,
        # Hard coded candidates
        # fixed_candidates=[(-0.4, -0.7), (-0.2, 0.5)],
        # fixed_candidates=[(-0.2, 0.5)],
        # debug_belief_task=(-0.2, 0.5),
        # debug_belief_task=None,
        # save_dir=f"{SAVE_ROOT}/191221_true",
        # save_dir=f"{SAVE_ROOT}/200110_test_eval",
        save_dir=f"{SAVE_ROOT}/200110_test_eval_all",
        exp_name="active_ird_exp_mid",
    )

    """ Experiment """
    # for ki in RANDOM_KEYS:
    #     key = random.PRNGKey(ki)
    #     experiment.update_key(key)
    #     experiment.run(TASK)

    """ Debug """
    for ki in RANDOM_KEYS:
        key = random.PRNGKey(ki)
        experiment.update_key(key)
        experiment.debug(f"{DEBUG_ROOT}/200110_test_eval_all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--GCP_MODE", action="store_true")
    args = parser.parse_args()

    # ENV_NAME = "Week3_02-v0"
    GCP_MODE = args.GCP_MODE
    ENV_NAME = "Week6_01-v0"
    TEST_MODE = False

    # Load parameters
    if not GCP_MODE:
        params = load_params("examples/acquisition_params.yaml")
    else:
        params = load_params("/dar_payload/rdb/examples/acquisition_params.yaml")
    locals().update(params)
    if not GCP_MODE:
        # RANDOM_KEYS = [1, 2, 3, 4]  # new macbook
        # RANDOM_KEYS = [1, 9, 10, 13]  # test
        # RANDOM_KEYS = [9, 10]  # test
        # RANDOM_KEYS = [13, 14, 15]
        # RANDOM_KEYS = [17]
        # RANDOM_KEYS = [18]
        # RANDOM_KEYS = [19]
        # RANDOM_KEYS = [20]
        # RANDOM_KEYS = [21]
        # RANDOM_KEYS = [22]
        # RANDOM_KEYS = [23]
        RANDOM_KEYS = [24]
        NUM_EVAL_WORKERS = 4
        # RANDOM_KEYS = [9, 10, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24]
        # NUM_EVAL_WORKERS = 8
        # RANDOM_KEYS = [9, 10, 11, 12] # alienware
        # NUM_EVAL_WORKERS = 4
        # RANDOM_KEYS = [13, 14, 15, 16] # dell
        # NUM_EVAL_WORKERS = 2
    main()
