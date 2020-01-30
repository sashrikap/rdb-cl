"""Debug Candidate
"""

from rdb.exps.active_ird import ExperimentActiveIRD
from rdb.exps.active import ActiveInfoGain, ActiveRatioTest, ActiveRandom
from rdb.infer.ird_oc import IRDOptimalControl
from rdb.optim.mpc import shooting_method
from rdb.distrib.particles import ParticleServer
from rdb.infer.utils import *
from rdb.exps.utils import *
from functools import partial
from jax import random
import numpyro.distributions as dist
import argparse


def debug_candidate(debug_dir, exp_name):
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

    ## Prior sampling & likelihood functions for PGM
    prior = LogUniformPrior(
        rng_key=None,
        normalized_key=NORMALIZED_KEY,
        feature_keys=FEATURE_KEYS,
        log_max=MAX_WEIGHT,
    )
    ird_proposal = IndGaussianProposal(
        rng_key=None,
        normalized_key=NORMALIZED_KEY,
        feature_keys=FEATURE_KEYS,
        proposal_var=IRD_PROPOSAL_VAR,
    )
    designer_proposal = IndGaussianProposal(
        rng_key=None,
        normalized_key=NORMALIZED_KEY,
        feature_keys=FEATURE_KEYS,
        proposal_var=DESIGNER_PROPOSAL_VAR,
    )

    ## Evaluation Server
    eval_server = ParticleServer(
        env_fn, controller_fn, num_workers=NUM_EVAL_WORKERS, parallel=PARALLEL
    )

    ird_model = IRDOptimalControl(
        rng_key=None,
        env_id=ENV_NAME,
        env_fn=env_fn,
        controller_fn=controller_fn,
        eval_server=eval_server,
        beta=BETA,
        true_w=TRUE_W,
        prior=prior,
        num_normalizers=NUM_NORMALIZERS,
        normalized_key=NORMALIZED_KEY,
        proposal=ird_proposal,
        sample_args={"num_warmups": NUM_WARMUPS, "num_samples": NUM_SAMPLES},
        designer_proposal=designer_proposal,
        designer_args={
            "num_warmups": NUM_DESIGNER_WARMUPS,
            "num_samples": NUM_DESIGNERS,
        },
        use_true_w=USER_TRUE_W,
        num_prior_tasks=NUM_PRIOR_TASKS,
        save_root=f"{SAVE_ROOT}/{SAVE_NAME}",
        exp_name=f"{EXP_NAME}",
    )

    """ Active acquisition function for experiment """
    active_fns = {
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
    keys = list(active_fns.keys())
    for key in keys:
        if key not in ACTIVE_FNS:
            del active_fns[key]

    SAVE_ROOT = data_dir()
    DEBUG_ROOT = data_dir()
    experiment = ExperimentActiveIRD(
        ird_model,
        active_fns,
        eval_server=eval_server,
        iterations=EXP_ITERATIONS,
        num_eval_tasks=NUM_EVAL_TASKS,
        num_eval_sample=NUM_EVAL_SAMPLES,
        num_eval_map=NUM_EVAL_MAP,
        num_active_tasks=NUM_ACTIVE_TASKS,
        num_active_sample=NUM_ACTIVE_SAMPLES,
        normalized_key=NORMALIZED_KEY,
        save_root=f"{SAVE_ROOT}/{SAVE_NAME}",
        exp_name=f"{EXP_NAME}",
        exp_params=PARAMS,
    )

    """ Experiment """
    for ki in CANDIDATE_KEYS:
        key = random.PRNGKey(ki)
        experiment.update_key(key)
        experiment.debug_candidate(DEBUG_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--GCP_MODE", action="store_true")
    args = parser.parse_args()

    # PARAMS = load_params("examples/params/active_template.yaml")
    PARAMS = load_params("examples/params/active_template.yaml")
    locals().update(PARAMS)

    DEBUG_DIR = "data/200120/"
    # EXP_NAME = "active_ird_exp_three"
    # ENV_NAME = "Week6_03-v0"
    EXP_NAME = "active_ird_exp_two"
    ENV_NAME = "Week6_02-v0"
    CANDIDATE_KEYS = list(range(8))

    # EXP_NAME = "interactive_ird_exp_three_random"
    # CANDIDATE_KEYS = [26]
    # ENV_NAME = "Week6_03-v0"

    # RAND_EXP_NAME = "random_ird_exp_mid"
    debug_candidate(DEBUG_DIR, EXP_NAME)
