"""Run Active-IRD Experiment Loop.

Note:
    * See (rdb.exps.active_ird.py) for more details.

"""
from rdb.exps.active_ird import ExperimentActiveIRD
from rdb.exps.active import ActiveInfoGain, ActiveRatioTest, ActiveRandom
from rdb.distrib.particles import ParticleServer
from rdb.infer.ird_oc import IRDOptimalControl
from rdb.optim.mpc import build_mpc
from functools import partial
from rdb.exps.utils import *
from rdb.infer import *
from jax import random
import numpyro.distributions as dist
import yaml, argparse, os
import copy
import ray


def main(random_key, evaluate=False):
    SAVE_ROOT = data_dir() if not GCP_MODE else "/gcp_output"  # Don'tchange this line
    DEBUG_ROOT = data_dir() if not GCP_MODE else "/gcp_input"

    # Define random key
    rng_key = random.PRNGKey(random_key)
    if evaluate:
        # Load pre-saved parameters and update
        print(
            f"\n======== Evaluating exp {EXP_NAME} from {SAVE_ROOT}/{SAVE_NAME}========\n"
        )
        params_path = f"{SAVE_ROOT}/{SAVE_NAME}/{EXP_NAME}/params_{str(rng_key)}.yaml"
        assert os.path.isfile(params_path)
        eval_params = load_params(params_path)
        locals().update(eval_params)

    def env_fn():
        import gym, rdb.envs.drive2d

        env = gym.make(ENV_NAME)
        env.reset()
        return env

    def controller_fn(env):
        controller, runner = build_mpc(
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
    weight_params = {"bins": HIST_BINS, "max_weights": MAX_WEIGHT}

    ird_model = IRDOptimalControl(
        rng_key=None,
        env_id=ENV_NAME,
        env_fn=env_fn,
        controller_fn=controller_fn,
        eval_server=eval_server,
        beta=BETA,
        true_w=TRUE_W,
        prior=prior,
        proposal=ird_proposal,
        num_normalizers=NUM_NORMALIZERS,
        sample_args={
            "num_warmups": NUM_WARMUPS,
            "num_samples": NUM_SAMPLES,
            "num_chains": NUM_IRD_CHAINS,
        },
        designer_proposal=designer_proposal,
        designer_args={
            "num_warmups": NUM_DESIGNER_WARMUPS,
            "num_samples": NUM_DESIGNERS,
            "num_chains": NUM_DESIGNER_CHAINS,
        },
        weight_params=weight_params,
        use_true_w=USER_TRUE_W,
        num_prior_tasks=NUM_PRIOR_TASKS,
        normalized_key=NORMALIZED_KEY,
        save_root=f"{SAVE_ROOT}/{SAVE_NAME}",
        exp_name=f"{EXP_NAME}",
    )

    ## Active acquisition function for experiment
    active_fns = {
        "infogain": ActiveInfoGain(
            rng_key=None, model=ird_model, beta=BETA, params=weight_params, debug=False
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

    ## Task sampling seed
    if FIXED_TASK_SEED is not None:
        fixed_task_seed = random.PRNGKey(FIXED_TASK_SEED)
    else:
        fixed_task_seed = None

    experiment = ExperimentActiveIRD(
        ird_model,
        active_fns,
        eval_server=eval_server,
        iterations=EXP_ITERATIONS,
        num_eval_tasks=NUM_EVAL_TASKS,
        num_eval_map=NUM_EVAL_MAP,
        num_active_tasks=NUM_ACTIVE_TASKS,
        num_active_sample=NUM_ACTIVE_SAMPLES,
        fixed_task_seed=fixed_task_seed,
        save_root=f"{SAVE_ROOT}/{SAVE_NAME}",
        exp_name=f"{EXP_NAME}",
        exp_params=PARAMS,
        normalized_key=NORMALIZED_KEY,
    )

    """ Experiment """
    experiment.update_key(rng_key)
    if evaluate:
        experiment.run_evaluation(override=True)
    else:
        experiment.run()
    ray.shutdown()  # Prepare for next run, which reinitialize ray with different seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--GCP_MODE", action="store_true")
    parser.add_argument("--EVALUATE", action="store_true")
    args = parser.parse_args()

    GCP_MODE = args.GCP_MODE
    EVALUATE = args.EVALUATE

    # Load parameters
    if not GCP_MODE:
        PARAMS = load_params("examples/params/active_template.yaml")
    else:
        PARAMS = load_params("/dar_payload/rdb/examples/params/active_params.yaml")
    locals().update(PARAMS)
    if not GCP_MODE:
        # RANDOM_KEYS = [24]
        NUM_EVAL_WORKERS = 4
    for ki in copy.deepcopy(RANDOM_KEYS):
        main(random_key=ki, evaluate=EVALUATE)


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
