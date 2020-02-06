"""Run Active-IRD Experiment Loop.

Note:
    * See (rdb.exps.active_ird.py) for more details.

"""
from rdb.exps.mcmc_convergence import ExperimentMCMC
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


def main(random_key):
    SAVE_ROOT = data_dir() if not GCP_MODE else "/gcp_output"  # Don't change this line
    DEBUG_ROOT = data_dir() if not GCP_MODE else "/gcp_input"

    # Define random key
    rng_key = random.PRNGKey(random_key)

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

    design_data = load_params(join(examples_dir(), f"designs/{ENV_NAME}.yaml"))
    assert design_data["ENV_NAME"] == ENV_NAME, "Environment name mismatch"

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
    # eval_server = ParticleServer(
    #     env_fn, controller_fn, num_workers=NUM_EVAL_WORKERS, parallel=PARALLEL
    # )
    eval_server = None
    weight_params = {"bins": HIST_BINS, "max_weights": MAX_WEIGHT}

    ird_model = IRDOptimalControl(
        rng_key=None,
        env_id=ENV_NAME,
        env_fn=env_fn,
        controller_fn=controller_fn,
        eval_server=eval_server,
        beta=BETA,
        true_w=None,
        prior=prior,
        proposal=ird_proposal,
        num_normalizers=NUM_NORMALIZERS,
        sample_args={
            "num_warmups": NUM_WARMUPS,
            "num_samples": NUM_SAMPLES,
            "num_chains": NUM_IRD_CHAINS,
            "use_dictlist": True,
        },
        designer_proposal=designer_proposal,
        designer_args={
            "num_warmups": NUM_DESIGNER_WARMUPS,
            "num_samples": NUM_DESIGNERS,
            "num_chains": NUM_DESIGNER_CHAINS,
            "use_dictlist": True,
        },
        weight_params=weight_params,
        normalized_key=NORMALIZED_KEY,
        save_root=f"{SAVE_ROOT}/{SAVE_NAME}",
        exp_name=f"{EXP_NAME}",
    )

    ## Task sampling seed
    if FIXED_TASK_SEED is not None:
        fixed_task_seed = random.PRNGKey(FIXED_TASK_SEED)
    else:
        fixed_task_seed = None

    NUM_DESIGNS = len(design_data["DESIGNS"])
    experiment = ExperimentMCMC(
        ird_model,
        eval_server=eval_server,
        num_eval_tasks=NUM_EVAL_TASKS,
        num_eval_map=NUM_EVAL_MAP,
        num_visualize_tasks=NUM_VISUALIZE_TASKS,
        fixed_task_seed=fixed_task_seed,
        design_data=design_data,
        save_root=f"{SAVE_ROOT}/{SAVE_NAME}",
        exp_name=f"{EXP_NAME}",
        exp_params=PARAMS,
        exp_mode=EXP_MODE,
        normalized_key=NORMALIZED_KEY,
    )
    """ Experiment """
    experiment.update_key(rng_key)
    experiment.run_designer()
    # ray.shutdown()  # Prepare for next run, which reinitialize ray with different seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--GCP_MODE", action="store_true")
    args = parser.parse_args()

    GCP_MODE = args.GCP_MODE

    # Load parameters
    if not GCP_MODE:
        PARAMS = load_params("examples/params/mcmc_template.yaml")
    else:
        PARAMS = load_params("/dar_payload/rdb/examples/params/mcmc_params.yaml")
    locals().update(PARAMS)
    if not GCP_MODE:
        RANDOM_KEYS = [23]
        NUM_EVAL_WORKERS = 4
    for ki in copy.deepcopy(RANDOM_KEYS):
        main(random_key=ki)
