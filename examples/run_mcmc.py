"""Run Active-IRD Experiment Loop.

Note:
    * See (rdb.exps.active_ird.py) for more details.

"""
import numpyro

numpyro.set_host_device_count(3)

from rdb.exps.mcmc_convergence import ExperimentMCMC
from rdb.exps.active import ActiveInfoGain, ActiveRatioTest, ActiveRandom
from rdb.distrib.particles import ParticleServer
from rdb.infer.ird_oc import IRDOptimalControl
from rdb.optim.mpc import build_mpc
from functools import partial
from rdb.exps.utils import *
from rdb.infer import *
from jax import random
import gym, rdb.envs.drive2d
import numpyro.distributions as dist
import yaml, argparse, os
import numpyro
import copy
import ray


# numpyro.enable_validation()


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

    def controller_fn(env, name=""):
        controller, runner = build_mpc(
            env,
            env.main_car.cost_runtime,
            dt=env.dt,
            replan=False,
            name=name,
            **CONTROLLER_ARGS,
        )
        return controller, runner

    design_data = load_params(join(examples_dir(), f"designs/{ENV_NAME}.yaml"))
    assert design_data["ENV_NAME"] == ENV_NAME, "Environment name mismatch"

    ## Prior sampling & likelihood functions for PGM
    def prior_fn(name=""):
        return LogUniformPrior(
            normalized_key=WEIGHT_PARAMS["normalized_key"],
            feature_keys=WEIGHT_PARAMS["feature_keys"],
            log_max=WEIGHT_PARAMS["max_weights"],
            name=name,
        )

    ## Evaluation Server
    # eval_server = ParticleServer(
    #     env_fn, controller_fn, num_workers=NUM_EVAL_WORKERS, parallel=PARALLEL
    # )
    eval_server = None
    designer = Designer(
        env_fn=env_fn,
        controller_fn=controller_fn,
        prior_fn=prior_fn,
        weight_params=WEIGHT_PARAMS,
        normalized_key=WEIGHT_PARAMS["normalized_key"],
        save_root=f"{SAVE_ROOT}/{SAVE_NAME}",
        **DESIGNER_ARGS,
    )

    ird_model = IRDOptimalControl(
        env_id=ENV_NAME,
        env_fn=env_fn,
        controller_fn=controller_fn,
        eval_server=eval_server,
        designer=designer,
        prior_fn=prior_fn,
        normalized_key=WEIGHT_PARAMS["normalized_key"],
        weight_params=WEIGHT_PARAMS,
        save_root=f"{SAVE_ROOT}/{SAVE_NAME}",
        **IRD_ARGS,
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
        normalized_key=WEIGHT_PARAMS["normalized_key"],
        fixed_task_seed=fixed_task_seed,
        design_data=design_data,
        save_root=f"{SAVE_ROOT}/{SAVE_NAME}",
        exp_params=PARAMS,
        **EXPERIMENT_ARGS,
    )
    """ Experiment """
    # with jax.disable_jit():
    experiment.update_key(rng_key)
    experiment.run_designer(DESIGNER_ARGS["exp_name"])
    # experiment.run_ird(IRD_ARGS["exp_name"])


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

    # max_chains = max([IRD_ARGS["sample_args"]["num_chains"], DESIGNER_ARGS["sample_args"]["num_chains"]])
    # numpyro.set_host_device_count(max_chains)

    for ki in copy.deepcopy(RANDOM_KEYS):
        main(random_key=ki)
